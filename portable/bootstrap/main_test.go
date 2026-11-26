package main

import (
	"archive/zip"
	"bytes"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func usePayload(t *testing.T, name string) {
	t.Helper()
	var data bytes.Buffer
	z := zip.NewWriter(&data)
	w, err := z.Create(name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write([]byte("test runtime")); err != nil {
		t.Fatal(err)
	}
	if err := z.Close(); err != nil {
		t.Fatal(err)
	}
	old := payload
	payload = data.Bytes()
	t.Cleanup(func() { payload = old })
}

func TestUnpackAndReuse(t *testing.T) {
	usePayload(t, "python/python.exe")
	root := t.TempDir()
	dir, err := unpack(root)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "python", "python.exe"))
	if err != nil || string(b) != "test runtime" {
		t.Fatal(err, string(b))
	}
	again, err := unpack(root)
	if err != nil || again != dir {
		t.Fatal(err, again)
	}
}

func TestConcurrentFirstLaunch(t *testing.T) {
	usePayload(t, "python/python.exe")
	root := t.TempDir()
	var group sync.WaitGroup
	for i := 0; i < 5; i++ {
		group.Add(1)
		go func() {
			defer group.Done()
			if _, err := unpack(root); err != nil {
				t.Error(err)
			}
		}()
	}
	group.Wait()
	items, err := os.ReadDir(filepath.Join(root, "boot"))
	if err != nil || len(items) != 1 {
		t.Fatal(err, len(items))
	}
}

func TestRejectOutsidePath(t *testing.T) {
	usePayload(t, "../escape")
	if _, err := unpack(t.TempDir()); err == nil {
		t.Fatal("accepted path outside release")
	}
}
