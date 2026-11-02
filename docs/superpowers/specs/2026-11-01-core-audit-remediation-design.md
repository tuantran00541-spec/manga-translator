# Core audit remediation

## Goal and constraints

Fix the verified defects from the audit and investigate the remaining reported risks without degrading translation recall. Work on an isolated branch from main. Keep the original and intermediate artifacts recoverable. Do not classify or exclude detected regions as watermarks before cleaning or translation: dialogue has previously been misclassified. Watermarks must still be detected and cleaned. Do not automatically treat an empty model response as proof that a region is a watermark or non-story text.

## Delivery sequence

1. Fix seam rendering that drops image changes, constrain all font-size paths, preserve separate lettering regions for double bubbles, and validate geometry against the available safe region. Add regression tests for seam order, numeric OCR sizes, bubble lobes, and text crossing the border. If the true bubble interior is unknown, avoid inventing a contour and flag uncertain layouts for review.
2. Preserve high-confidence detections with empty segmentation as reviewable candidates without erasing an unverified rectangle. Address missed fragments, duplicated model inference, unsafe merging and missed-text retries. Do not suppress watermark detections.
3. Investigate destructive cleaning using original, clean, masks and per-stage evidence. Protect art and bubble outlines, compare native-resolution or tiled inpainting on representative crops, and verify both text coverage and damage outside target regions. Avoid changing defaults solely from synthetic results.
4. Improve AI input readability and box annotation, distinguish model omissions from explicit empty responses without classifying them as watermark, and preserve untranslated content for human review. Address ordering of chapter context, missing scan transcripts, language-specific series memory, budget behavior and accurate partial-failure reporting. Preserve compatibility with existing stored chapters.
5. Fix verified gateway reservation and ledger races, local API request authentication and CSRF boundaries, browser downloader network isolation, and bounded skill archive reads. Preserve desktop and extension workflows and avoid silently changing externally exposed service behavior.

## Validation

Each verified defect gets a failing regression test before implementation. Test related components after each change and run the broader available suite at integration points. Re-run both Blade of the Discarded chapters through GitHub Actions after the fixes; compare original, clean, and rendered outputs, especially chapter 2 slices 5, 9 and 15 and chapter 1 slice 16. Check missing dialogue, bubble placement, art damage, watermark cleaning, export blockers, completeness of render artifacts, cost, and CI outcome. Report what was and was not verified, and do not rely on the current scorecard as a visual-quality oracle.

## Safety and scope

Keep main unchanged. Do not push or merge without explicit user authorization. Use existing model configuration and do not expose secrets. Avoid committing copyrighted chapter images or audit output to a code branch. Preserve backward compatibility for series memory and existing job data, or provide an explicit safe migration. Any proposed change that cannot be validated safely is reported separately rather than represented as fixed.
