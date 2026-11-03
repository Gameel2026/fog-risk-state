#!/bin/sh
# Cortex-M4F build with LLVM (clang 18, ld.lld). Outputs: fog_state.elf, flash.bin, symbols.txt, size.txt, fog_state.su
CF="--target=thumbv7em-none-eabihf -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 -ffreestanding -fno-builtin -fno-math-errno -ffp-contract=off -nostdlib -Wall"
clang $CF -fstack-usage -c fog_state.c -o fog_state.o
ld.lld -T link.ld --entry=run -o fog_state.elf fog_state.o
llvm-objcopy -O binary --only-section=.text --only-section=.rodata fog_state.elf flash.bin
llvm-nm fog_state.elf > symbols.txt
llvm-size -A fog_state.elf > size.txt
