# Images Folder Guidelines

File naming convention for blog images (Jekyll / GitHub Pages):

Format: `YYYY-MM-DD-topic-slug[-detail][-vN].ext`

Components:
- YYYY-MM-DD: Date of related post (use post date)
- topic-slug: lowercase, hyphen-separated, concise (e.g., transformer-study7, attention, embedding)
- detail (optional): short purpose (e.g., tsne, architecture, pipeline, loss-curve)
- vN (optional): version number if updated (v1, v2 ...)

Examples:
- 2025-08-29-transformer-study7-tsne-v1.png
- 2025-08-29-transformer-study7-architecture.png
- 2025-08-29-embedding-space-overview-v2.webp

General rules:
1. Use only lowercase letters, digits, and hyphens.
2. Prefer `.webp` for diagrams/screenshots (smaller), `.png` for lossless vector/raster with transparency, `.jpg` for photos.
3. Keep width under ~1600px unless detail requires more. Optimize (<300KB when possible).
4. Avoid spaces—never rely on URL encoding in markdown.
5. If replacing an image but want cache-bust: increment the version suffix instead of overwriting.
6. Add alt text describing content; avoid purely decorative images unless necessary.

Optional front-matter usage:
You can reference images with `{{ '/assets/images/<filename>' | relative_url }}` for portability.

Optimization tip (macOS/Linux example):
```
# png to webp
cwebp -q 85 input.png -o output.webp
```

If many images accumulate, consider subfolders by year or topic, but keep paths shallow for simpler includes.
