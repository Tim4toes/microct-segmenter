// The bootstrap only unpacks the embedded private interpreter and starts the release manager.
package main

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
)

// The release builder supplies this archive via go:embed.
var payload []byte

var releaseVersion = "development"

var reportError = func(message string) { fmt.Fprintln(os.Stderr, message) }

func unpack(root string) (string, error) {
	id := fmt.Sprintf("%x", sha256.Sum256(payload))
	dir := filepath.Join(root, "boot", id[:16])
	ready := func() bool {
		b, err := os.ReadFile(filepath.Join(dir, "complete"))
		return err == nil && string(b) == id
	}
	if ready() {
		return dir, nil
	}
	if err := os.MkdirAll(filepath.Dir(dir), 0700); err != nil {
		return "", err
	}
	stage, err := os.MkdirTemp(filepath.Dir(dir), "unpack-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(stage)
	reader, err := zip.NewReader(bytes.NewReader(payload), int64(len(payload)))
	if err != nil {
		return "", err
	}
	fmt.Println("Preparing the private Python runtime...")
	for _, entry := range reader.File {
		name := filepath.FromSlash(entry.Name)
		if !filepath.IsLocal(name) || !entry.Mode().IsRegular() {
			return "", fmt.Errorf("unexpected payload entry %q", name)
		}
		target := filepath.Join(stage, name)
		if err := os.MkdirAll(filepath.Dir(target), 0700); err != nil {
			return "", err
		}
		if err := extractFile(entry, target); err != nil {
			return "", err
		}
	}
	if err := os.WriteFile(filepath.Join(stage, "complete"), []byte(id), 0600); err != nil {
		return "", err
	}
	if err := os.Rename(stage, dir); err != nil && !ready() {
		return "", err
	}
	return dir, nil
}

func extractFile(entry *zip.File, target string) error {
	input, err := entry.Open()
	if err != nil {
		return err
	}
	defer input.Close()
	out, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(out, input)
	closeErr := out.Close()
	return errors.Join(copyErr, closeErr)
}

func run() error {
	if len(os.Args) == 2 && os.Args[1] == "--version" {
		fmt.Println("MicroCT Segmenter launcher", releaseVersion)
		return nil
	}
	userDir, err := os.UserCacheDir()
	if err != nil {
		return err
	}
	root := filepath.Join(userDir, "MicroCTSegmenter")
	dir, err := unpack(root)
	if err != nil {
		return err
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	args := []string{"-I", "-u", filepath.Join(dir, "manage.py"), root, exe}
	cmd := exec.Command(filepath.Join(dir, "python", "python.exe"), append(args, os.Args[1:]...)...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}

func main() {
	if err := run(); err != nil {
		var status *exec.ExitError
		if errors.As(err, &status) && status.ExitCode() == 1 {
			os.Exit(1) // The manager already displayed its diagnostic.
		}
		reportError(err.Error())
		os.Exit(1)
	}
}
