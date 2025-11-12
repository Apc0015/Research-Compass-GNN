# Quick Reference: Multi-Format Upload

## Supported Formats

| Type | Extensions | Example |
|------|------------|---------|
| PDF | `.pdf` | `paper.pdf` |
| Text | `.txt`, `.md`, `.text` | `notes.txt` |
| Word | `.docx`, `.doc` | `paper.docx` |
| Web | `.html`, `.htm` | `article.html` |
| XML | `.xml` | `document.xml` |
| Archive | `.tar`, `.tar.gz`, `.tgz`, `.zip` | `papers.tar.gz` |

## URL Formats

```
# arXiv
https://arxiv.org/abs/1706.03762
https://arxiv.org/pdf/2010.11929.pdf
1706.03762

# DOI
https://doi.org/10.1145/3292500.3330989
10.1145/3292500.3330989

# Direct PDF
https://example.com/paper.pdf
```

## Quick Start

### 1. Launch UI
```bash
python scripts/launcher.py
```

### 2. Upload Options

**Option A: Upload Files**
- Drag & drop PDFs, DOCX, TXT, HTML, XML
- Or click to browse

**Option B: Add URLs**
- One per line or comma-separated
- Supports arXiv, DOI, direct links

**Option C: Upload Archive**
- Upload .tar.gz or .zip
- Auto-extracts all papers

**Option D: Mix All**
- Combine files + URLs + archives
- Process together in one graph

### 3. Process
Click "🔄 Process Papers & Build Graph"

### 4. Train
Select model → Set epochs → Click "🚀 Train"

## Configuration

Edit `config.yaml`:

```yaml
upload:
  max_file_size: 104857600    # 100MB
  max_archive_files: 200      # Max files per archive
  download_timeout: 30        # URL timeout (sec)
  
  allowed_domains:
    - 'arxiv.org'
    - 'doi.org'
    - 'aclanthology.org'
```

## Common Tasks

### Task 1: Upload Local PDFs
1. Click file upload
2. Select multiple PDFs
3. Click "Process"

### Task 2: Download from arXiv
1. Paste URLs in text box:
   ```
   https://arxiv.org/abs/1706.03762
   https://arxiv.org/abs/2010.11929
   ```
2. Click "Process"

### Task 3: Batch Upload Archive
1. Create `papers.zip` with PDFs
2. Upload the ZIP file
3. System extracts all papers
4. Click "Process"

### Task 4: Mixed Sources
1. Upload `local.pdf`
2. Add URL `https://arxiv.org/abs/1706.03762`
3. Upload `more.tar.gz`
4. Click "Process" → All combined!

## Troubleshooting

**Q: File upload fails**
- Check file size < 100MB
- Verify format is supported
- Try renaming file if special characters

**Q: URL download fails**
- Check internet connection
- Verify URL is from allowed domain
- Try direct arXiv link instead of abs page

**Q: Archive extraction fails**
- Check archive < 200 files
- Verify archive not corrupted
- Try extracting manually first

**Q: No citations found**
- Citations are regex-based (limited)
- Works best with standard formats
- Try PDF with proper citations

## Tips

✅ **Do:**
- Use standard citation formats
- Keep file sizes reasonable
- Use trusted academic sources
- Combine related papers

❌ **Don't:**
- Upload scanned PDFs (no OCR)
- Use untrusted domains
- Exceed file size limits
- Upload very large archives

## Examples

See `examples/demo_multi_format_upload.py` for code examples.

See `docs/MULTI_FORMAT_UPLOAD.md` for complete documentation.
