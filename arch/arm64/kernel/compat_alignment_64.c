#include <linux/kernel.h>
#include <linux/ktime.h>
#include <linux/timekeeping.h>
#include <linux/uaccess.h>
#include <linux/slab.h>

#include <asm/fpsimd.h>
#include <asm/neon.h>
#include <asm/simd.h>
#include <asm/ptrace.h>
#include <asm/traps.h>
#include <asm/cacheflush.h>
#include <asm/barrier.h>

#include <generated/asm/sysreg-defs.h>

/*
 * ARM64 Alignment Fault Handler - Optimized for ARM64 SBCs
 * 
 * Optimizations for Raspberry Pi 5 and similar ARM64 platforms:
 * - Cache-friendly memory access patterns
 * - Reduced function call overhead with inlining
 * - Optimized SIMD operations with batch processing
 * - ARM64-native bit manipulation operations
 * - Efficient branch prediction hints
 * - Streamlined error paths
 */

/* Fault descriptor optimized for cache efficiency */
struct fault_desc {
	void __user *addr;       /* Target memory address */
	u32 instr;              /* Faulting instruction */
	
	/* Register information - packed for cache efficiency */
	u8 reg1;                /* Primary target register */
	u8 reg2;                /* Secondary register (for pairs) */
	u8 base_reg;            /* Base address register */
	u8 offset_reg;          /* Offset register (register addressing) */
	
	/* Operation parameters */
	u16 width_bits;         /* Access width in bits */
	u16 extend_width;       /* Sign extension target width */
	s16 immediate;          /* Immediate offset value */
	
	/* Flags - packed into single byte for cache efficiency */
	u8 is_load:1;           /* 1=load, 0=store */
	u8 is_simd:1;           /* SIMD/vector operation */
	u8 is_pair:1;           /* Load/store pair */
	u8 sign_extend:1;       /* Sign extend on load */
	u8 scale_offset:1;      /* Scale offset by access size */
	u8 post_index:1;        /* Post-increment addressing */
	u8 pre_index:1;         /* Pre-increment addressing */
	u8 reserved:1;          /* Reserved for future use */
};

/* ARM64-optimized instruction fetch */
static __always_inline int get_fault_instruction(struct pt_regs *regs, u32 *instr)
{
	__le32 le_instr;
	int ret = get_user(le_instr, (__le32 __user *)instruction_pointer(regs));
	if (unlikely(ret))
		return ret;
	*instr = __le32_to_cpu(le_instr);
	return 0;
}

/* Sign extension helper - optimized for common cases */
static __always_inline s64 sign_extend_imm(u64 value, unsigned int bits)
{
	if (unlikely(bits == 0 || bits > 64))
		return 0;
	
	if (likely(bits <= 32)) {
		u32 mask = (1U << (bits - 1));
		return (s32)((value ^ mask) - mask);
	}

	u64 mask = (1ULL << (bits - 1));
	return (s64)((value ^ mask) - mask);
}

/* Optimized SIMD access with reduced kernel_neon overhead */
static __always_inline int simd_access_reg(u8 reg, u64 data[2], bool write)
{
	if (unlikely(!may_use_simd() || reg >= 32))
		return -EINVAL;

	kernel_neon_begin();
	
	struct user_fpsimd_state *fpsimd = &current->thread.uw.fpsimd_state;
	__uint128_t *vreg = (__uint128_t *)&fpsimd->vregs[reg];
	
	if (write) {
		*vreg = ((__uint128_t)data[1] << 64) | data[0];
	} else {
		__uint128_t reg_value = *vreg;
		data[0] = (u64)reg_value;
		data[1] = (u64)(reg_value >> 64);
	}

	kernel_neon_end();
	return 0;
}

/* Batch SIMD operations for pairs - reduces kernel_neon overhead */
static __always_inline int simd_access_pair(u8 reg1, u8 reg2, u64 data1[2], u64 data2[2], bool write)
{
	if (unlikely(!may_use_simd() || reg1 >= 32 || reg2 >= 32))
		return -EINVAL;

	kernel_neon_begin();
	
	struct user_fpsimd_state *fpsimd = &current->thread.uw.fpsimd_state;
	__uint128_t *vregs = (__uint128_t *)fpsimd->vregs;
	
	if (write) {
		vregs[reg1] = ((__uint128_t)data1[1] << 64) | data1[0];
		vregs[reg2] = ((__uint128_t)data2[1] << 64) | data2[0];
	} else {
		__uint128_t reg1_value = vregs[reg1];
		__uint128_t reg2_value = vregs[reg2];
		
		data1[0] = (u64)reg1_value;
		data1[1] = (u64)(reg1_value >> 64);
		data2[0] = (u64)reg2_value;
		data2[1] = (u64)(reg2_value >> 64);
	}

	kernel_neon_end();
	return 0;
}

/* Unified memory transfer optimized for ARM64 cache line efficiency */
static __always_inline int transfer_data(void __user *addr, u8 *data, unsigned int size, bool to_user)
{
	unsigned int remaining = size;

	if (unlikely(size == 0 || size > 16))
		return -EINVAL;
	
	/* ARM64 prefers 64-bit aligned accesses - optimize for cache lines */
	while (remaining >= 8 && IS_ALIGNED((unsigned long)addr, 8)) {
		int ret = to_user ? 
			put_user(*(u64 *)data, (u64 __user *)addr) :
			get_user(*(u64 *)data, (u64 __user *)addr);
		if (unlikely(ret))
			return ret;
		data += 8;
		addr += 8;
		remaining -= 8;
	}
	
	/* 32-bit accesses for remainder */
	while (remaining >= 4 && IS_ALIGNED((unsigned long)addr, 4)) {
		int ret = to_user ?
			put_user(*(u32 *)data, (u32 __user *)addr) :
			get_user(*(u32 *)data, (u32 __user *)addr);
		if (unlikely(ret))
			return ret;
		data += 4;
		addr += 4;
		remaining -= 4;
	}

	/* 16-bit accesses */
	while (remaining >= 2 && IS_ALIGNED((unsigned long)addr, 2)) {
		int ret = to_user ?
			put_user(*(u16 *)data, (u16 __user *)addr) :
			get_user(*(u16 *)data, (u16 __user *)addr);
		if (unlikely(ret))
			return ret;
		data += 2;
		addr += 2;
		remaining -= 2;
	}
	
	/* Handle remaining bytes */
	while (remaining > 0) {
		int ret = to_user ?
			put_user(*data, (u8 __user *)addr) :
			get_user(*data, (u8 __user *)addr);
		if (unlikely(ret))
			return ret;
		data++;
		addr++;
		remaining--;
	}
	
	return 0;
}

/* ARM64-optimized block clear using 64-bit stores */
static int clear_user_block(void __user *addr, u64 size)
{
	u64 remaining = size;
	const u64 zero = 0;
	
	/* Use 64-bit writes for optimal ARM64 cache line utilization */
	while (remaining >= 8) {
		if (put_user(zero, (u64 __user *)addr))
			return -EFAULT;
		addr += 8;
		remaining -= 8;
	}
	
	/* Handle remaining bytes */
	while (remaining > 0) {
		if (put_user(0, (u8 __user *)addr))
			return -EFAULT;
		addr++;
		remaining--;
	}
	
	return 0;
}

/* Extended register computation for register offset addressing */
static __always_inline u64 compute_extended_offset(u64 reg_val, u8 extend_type, u8 shift)
{
	u64 result;
	bool is_signed = (extend_type & 0x4) != 0;
	bool is_64bit = (extend_type & 0x1) != 0;
	
	if (likely(!is_signed)) {
		/* Zero extension */
		result = is_64bit ? reg_val : (reg_val & 0xFFFFFFFF);
	} else {
		/* Sign extension */
		if (is_64bit) {
			result = reg_val;  /* Already 64-bit */
		} else {
			/* Sign extend from 32-bit */
			result = (u64)(s64)(s32)(reg_val & 0xFFFFFFFF);
		}
	}
	
	return result << shift;
}

/* Streamlined load/store execution with optimized error paths */
static int execute_ls_operation(struct pt_regs *regs, const struct fault_desc *desc)
{
	int ret = 0;
	u64 reg_data[2] = {0, 0};
	u64 reg2_data[2] = {0, 0};
	unsigned int byte_count = desc->width_bits >> 3;
	
	if (!desc->is_load) {
		/* Store operation - read from registers */
		if (desc->is_simd) {
			ret = desc->is_pair ?
				simd_access_pair(desc->reg1, desc->reg2, reg_data, reg2_data, false) :
				simd_access_reg(desc->reg1, reg_data, false);
			if (unlikely(ret))
				return ret;
		} else {
			reg_data[0] = regs->regs[desc->reg1];
			if (desc->is_pair)
				reg2_data[0] = regs->regs[desc->reg2];
		}
		
		/* Memory writes */
		ret = transfer_data(desc->addr, (u8 *)reg_data, byte_count, true);
		if (unlikely(ret))
			return ret;

		if (desc->is_pair) {
			ret = transfer_data(desc->addr + byte_count, (u8 *)reg2_data, byte_count, true);
			if (unlikely(ret))
				return ret;
		}
	} else {
		/* Load operation - read from memory */
		ret = transfer_data(desc->addr, (u8 *)reg_data, byte_count, false);
		if (unlikely(ret))
			return ret;

		if (desc->is_pair) {
			ret = transfer_data(desc->addr + byte_count, (u8 *)reg2_data, byte_count, false);
			if (unlikely(ret))
				return ret;
		}

		/* Handle sign extension for scalar loads */
		if (desc->sign_extend && !desc->is_simd && desc->extend_width > desc->width_bits) {
			u64 sign_bit = 1ULL << (desc->width_bits - 1);
			if (reg_data[0] & sign_bit && desc->extend_width <= 64) {
				u64 extend_mask = ~((1ULL << desc->width_bits) - 1);
				reg_data[0] |= extend_mask;
			}
		}

		/* Register writes */
		if (desc->is_simd) {
			ret = desc->is_pair ?
				simd_access_pair(desc->reg1, desc->reg2, reg_data, reg2_data, true) :
				simd_access_reg(desc->reg1, reg_data, true);
			if (unlikely(ret))
				return ret;
		} else {
			regs->regs[desc->reg1] = reg_data[0];
			if (desc->is_pair)
				regs->regs[desc->reg2] = reg2_data[0];
		}
	}

	/* Handle address writeback for pre/post-indexed addressing */
	if (desc->pre_index || desc->post_index) {
		regs->regs[desc->base_reg] += desc->immediate;
	}

	arm64_skip_faulting_instruction(regs, 4);
	return 0;
}

/* Load/store pair decoder - optimized addressing mode handling */
static int decode_ls_pair(u32 instr, struct pt_regs *regs, struct fault_desc *desc)
{
	u8 opc = (instr >> 30) & 3;
	u8 addressing = (instr >> 23) & 3;
	u8 load = (instr >> 22) & 1;
	u8 simd = (instr >> 26) & 1;
	u16 imm7 = (instr >> 15) & 0x7f;
	u8 Rt2 = (instr >> 10) & 0x1f;
	u8 Rn = (instr >> 5) & 0x1f;
	u8 Rt = instr & 0x1f;
	
	if (addressing > 3)
		return -EINVAL;
	
	s64 offset = sign_extend_imm(imm7, 7);
	
	desc->is_load = load;
	desc->is_simd = simd;
	desc->is_pair = 1;
	desc->reg1 = Rt;
	desc->reg2 = Rt2;
	desc->base_reg = Rn;
	desc->immediate = offset;
	desc->post_index = (addressing == 1);
	desc->pre_index = (addressing == 3);

	if (simd) {
		desc->width_bits = 32 << opc;
		offset <<= (2 + opc);  /* Scale by access size */
	} else {
		/* Scalar pairs */
		switch (opc) {
		case 0: desc->width_bits = 32; offset <<= 2; break;
		case 2: desc->width_bits = 64; offset <<= 3; break;
		default:
			printk("Invalid scalar pair opc=%u in instr=0x%08x\n", opc, instr);
			return -EINVAL;
		}
	}
	
	/* Address calculation with proper indexing */
	desc->addr = (void __user *)(regs->regs[Rn] + 
		(desc->pre_index ? offset : (desc->post_index ? 0 : offset)));
	
	return 0;
}

/* Load/store register with unsigned immediate decoder */
static int decode_ls_unsigned_imm(u32 instr, struct pt_regs *regs, struct fault_desc *desc)
{
	u8 size = (instr >> 30) & 3;
	u8 simd = (instr >> 26) & 1;
	u8 opc = (instr >> 22) & 3;
	u16 imm12 = (instr >> 10) & 0xfff;
	u8 Rn = (instr >> 5) & 0x1f;
	u8 Rt = instr & 0x1f;
	
	u8 load = opc & 1;
	u8 width_shift;
	
	if (simd) {
		width_shift = size | ((opc & 2) << 1);
		/* Invalid size/opc combination for SIMD */
		if ((size & 1) && (opc & 2))
			return -EINVAL;
		desc->sign_extend = 0;
	} else {
		width_shift = size;
		desc->sign_extend = (opc & 2) >> 1;
		desc->extend_width = desc->sign_extend ? 32 : 64;
	}
	
	desc->is_load = load;
	desc->is_simd = simd;
	desc->is_pair = 0;
	desc->width_bits = 8 << width_shift;
	desc->reg1 = Rt;
	desc->base_reg = Rn;
	desc->addr = (void __user *)(regs->regs[Rn] + (imm12 << width_shift));
	
	return 0;
}

/* Load/store register offset decoder */
static int decode_ls_reg_offset(u32 instr, struct pt_regs *regs, struct fault_desc *desc)
{
	u8 size = (instr >> 30) & 3;
	u8 simd = (instr >> 26) & 1;
	u8 opc = (instr >> 22) & 3;
	u8 Rm = (instr >> 16) & 0x1f;
	u8 extend_type = (instr >> 13) & 7;
	u8 scale = (instr >> 12) & 1;
	u8 Rn = (instr >> 5) & 0x1f;
	u8 Rt = instr & 0x1f;
	
	u8 load = opc & 1;
	u8 width_shift = simd ? (size | ((opc & 2) << 1)) : size;
	u8 shift = scale ? width_shift : 0;
	
	desc->is_load = load;
	desc->is_simd = simd;
	desc->is_pair = 0;
	desc->width_bits = 8 << width_shift;
	desc->reg1 = Rt;
	desc->base_reg = Rn;
	desc->offset_reg = Rm;
	desc->scale_offset = scale;
	
	if (!simd) {
		desc->sign_extend = (opc & 2) >> 1;
		desc->extend_width = desc->sign_extend ? 32 : 64;
	}
	
	u64 offset = compute_extended_offset(regs->regs[Rm], extend_type, shift);
	desc->addr = (void __user *)(regs->regs[Rn] + offset);
	
	return 0;
}

/* Load/store unscaled immediate decoder */
static int decode_ls_unscaled_imm(u32 instr, struct pt_regs *regs, struct fault_desc *desc)
{
	u8 size = (instr >> 30) & 3;
	u8 simd = (instr >> 26) & 1;
	u8 opc = (instr >> 22) & 3;
	u16 imm9 = (instr >> 12) & 0x1ff;
	u8 Rn = (instr >> 5) & 0x1f;
	u8 Rt = instr & 0x1f;
	
	s16 offset = sign_extend_imm(imm9, 9);
	u8 load = opc & 1;
	
	desc->is_load = load;
	desc->is_simd = simd;
	desc->is_pair = 0;
	desc->reg1 = Rt;
	desc->base_reg = Rn;
	desc->immediate = offset;
	desc->addr = (void __user *)(regs->regs[Rn] + offset);
	
	if (simd) {
		desc->width_bits = 8 << (size | ((opc & 2) << 1));
		desc->sign_extend = 0;
	} else {
		desc->width_bits = 8 << size;
		desc->sign_extend = (opc & 2) >> 1;
		desc->extend_width = desc->sign_extend ? 32 : 64;
	}
	
	return 0;
}

/* Compare-and-swap atomic operation handler */
static int handle_cas_operation(u32 instr, struct pt_regs *regs)
{
	u8 size = (instr >> 30) & 3;
	u8 Rs = (instr >> 16) & 0x1f;
	u8 Rt2 = (instr >> 10) & 0x1f;
	u8 Rn = (instr >> 5) & 0x1f;
	u8 Rt = instr & 0x1f;
	
	if (Rt2 != 0x1f)  /* Must be single register CAS */
		return -EINVAL;
	
	unsigned int width_bits = 8 << size;
	unsigned int byte_count = width_bits >> 3;
	void __user *addr = (void __user *)regs->regs[Rn];
	u64 compare_val = regs->regs[Rs] & ((1ULL << width_bits) - 1);
	u64 new_val = regs->regs[Rt] & ((1ULL << width_bits) - 1);
	
	/* Read current value */
	u64 current_val = 0;
	int ret = transfer_data(addr, (u8 *)&current_val, byte_count, false);
	if (ret)
		return ret;
	
	current_val &= (1ULL << width_bits) - 1;

	printk("CAS operation not atomic at %px, size %d bits\n", addr, width_bits);
	
	if (current_val == compare_val) {
		/* Values match, perform the swap */
		ret = transfer_data(addr, (u8 *)&new_val, byte_count, true);
		if (ret)
			return ret;
	}
	
	/* Always write back the original value that was read */
	regs->regs[Rs] = current_val;
	arm64_skip_faulting_instruction(regs, 4);

	return 0;
}

/* Optimized instruction classifier with ARM64-specific patterns */
static int handle_ls_instruction(u32 instr, struct pt_regs *regs)
{
	struct fault_desc desc = {0};
	desc.instr = instr;

	u8 op0 = (instr >> 28) & 0xf;
	u8 op1 = (instr >> 26) & 1;
	u8 op2 = (instr >> 23) & 3;
	u8 op3 = (instr >> 16) & 0x3f;
	u8 op4 = (instr >> 10) & 3;
	int ret = -EINVAL;

	// TODO: remove after debugging
	printk("Handling Load/Store instruction: op0=0x%x op1=0x%x op2=0x%x op3=0x%x op4=0x%x instr=0x%08x\n",
	       op0, op1, op2, op3, op4, instr);

	/* Optimize for most common cases first - better branch prediction */
	if (likely((op0 & 3) == 2)) {
		/* Load/store pairs - most common for alignment faults */
		ret = decode_ls_pair(instr, regs, &desc);
		if (likely(!ret))
			ret = execute_ls_operation(regs, &desc);
	} else if ((op0 & 3) == 3 && (op2 & 2) == 2) {
		/* Unsigned immediate - second most common */
		ret = decode_ls_unsigned_imm(instr, regs, &desc);
		if (likely(!ret))
			ret = execute_ls_operation(regs, &desc);
	} else if ((op0 & 3) == 3 && (op2 & 2) == 0 && (op3 & 0x20) == 0x20 && op4 == 2) {
		/* Register offset */
		ret = decode_ls_reg_offset(instr, regs, &desc);
		if (likely(!ret))
			ret = execute_ls_operation(regs, &desc);
	} else if ((op0 & 3) == 3 && (op2 & 2) == 0 && (op3 & 0x20) == 0x00 && op4 == 0) {
		/* Unscaled immediate - handles 0x3c80c03f pattern */
		ret = decode_ls_unscaled_imm(instr, regs, &desc);
		if (likely(!ret))
			ret = execute_ls_operation(regs, &desc);
	} else if ((op0 & 3) == 0 && op1 == 0 && op2 == 1 && (op3 & 0x20) == 0x20) {
		/* Compare-and-swap */
		ret = handle_cas_operation(instr, regs);
	} else if (op0 == 0xf && op1 == 0 && op2 == 0 && op3 == 0 && op4 == 1) {
		/* Handle 0xf8008404 pattern */
		ret = decode_ls_reg_offset(instr, regs, &desc);
		if (likely(!ret))
			ret = execute_ls_operation(regs, &desc);
	}

	return ret;
}

/* System instruction handler - DC ZVA optimized for ARM64 cache architecture */
static int handle_system_instruction(u32 instr, struct pt_regs *regs)
{
	u8 op1 = (instr >> 16) & 0x7;
	u8 op2 = (instr >> 5) & 0x7;
	u8 CRn = (instr >> 12) & 0xf;
	u8 CRm = (instr >> 8) & 0xf;
	bool L = (instr >> 21) & 1;
	u8 Rt = instr & 0x1f;
 
	if (!L && op1 == 0x3 && op2 == 1 && CRn == 0x7 && CRm == 4) {
		/* DC ZVA - optimized for ARM64 cache line efficiency */
		u64 dczid_el0 = read_sysreg_s(SYS_DCZID_EL0);

		if (unlikely((dczid_el0 >> DCZID_EL0_DZP_SHIFT) & 1))
			return -EINVAL;

		u16 block_size = 4 << (dczid_el0 & 0xf);
		void __user *addr = (void __user *)regs->regs[Rt];
		void __user *aligned_addr = (void __user *)((unsigned long)addr & ~(block_size - 1));

		int ret = clear_user_block(aligned_addr, block_size);
		if (likely(!ret)) {
			/* Memory barrier for cache coherency */
			dsb(sy);
			arm64_skip_faulting_instruction(regs, 4);
		}

		return ret;
	}
	
	return -EINVAL;
}

/* Branch/Exception/System instruction classifier */
static int handle_branch_except_system(u32 instr, struct pt_regs *regs)
{
	u8 op0 = (instr >> 29) & 0x7;
	u32 op1 = (instr >> 5) & 0x1fffff;

	// TODO: remove after debugging
	// printk("Handling Branch/Exception/System instruction: op0=0x%x op1=0x%x instr=0x%08x\n",
	//        op0, op1, instr);

	if (op0 == 0x6 && (op1 & 0x1ec000) == 0x84000) {
		return handle_system_instruction(instr, regs);
	}
	
	// TODO: use `pr_info_ratelimited` after debugging
	printk("Unhandled Branch/Exception/System: op0=0x%x op1=0x%x instr=0x%08x\n",
			   op0, op1, instr);
	return -EINVAL;
}

/* Main alignment fault handler - optimized for ARM64 instruction classification */
int do_alignment_fixup(unsigned long addr, struct pt_regs *regs)
{
	u32 instr = 0;
	int ret = get_fault_instruction(regs, &instr);
	if (unlikely(ret)) {
		// TODO: use `pr_debug` after debugging
		printk("Failed to fetch faulting instruction at PC=0x%lx\n", 
						instruction_pointer(regs));
		return 1;
	}

	/* ARM64-optimized instruction classification using bit patterns */
	u8 op0 = (instr >> 25) & 0x1f;

	/* Fast path for load/store instructions - most common case */
	if (likely((op0 & 0x5) == 0x4)) {
		ret = handle_ls_instruction(instr, regs);
		if (unlikely(ret)) {
			// TODO: use `pr_debug` after debugging
			printk("Load/Store fixup failed: instr=0x%08x PC=0x%lx ret=%d\n",
				instr, instruction_pointer(regs), ret);
		}
		return ret;
	} else if ((op0 & 0xe) == 0xa) {
		/* Branch/Exception/System instructions */
		return handle_branch_except_system(instr, regs);
	} else {
		/* Unsupported instruction type */
		// TODO: use `pr_debug` after debugging
		printk("Unsupported alignment fault: op0=0x%x instr=0x%08x PC=0x%lx\n",
				   op0, instr, instruction_pointer(regs));
		return -EINVAL;
	}
}
