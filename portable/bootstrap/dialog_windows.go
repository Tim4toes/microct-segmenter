package main

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

func init() { reportError = windowsError }

func windowsError(message string) {
	fmt.Fprintln(os.Stderr, message)
	title, _ := syscall.UTF16PtrFromString("MicroCT Segmenter")
	body, _ := syscall.UTF16PtrFromString(message)
	syscall.NewLazyDLL("user32.dll").NewProc("MessageBoxW").Call(0,
		uintptr(unsafe.Pointer(body)), uintptr(unsafe.Pointer(title)), 0x10)
}
