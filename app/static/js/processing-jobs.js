(() => {
  const PROCESS_JOB_POLL_MS = 700;

  function processingButtons() {
    return Array.from(document.querySelectorAll(".preview-primary-action, #start-action"));
  }

  function setProcessingUi(snapshot) {
    const total = Number(snapshot?.total) || 0;
    const completed = Number(snapshot?.completed) || 0;
    const active = snapshot?.status === "pending" || snapshot?.status === "running";
    processingButtons().forEach((btn) => {
      if (!btn || !btn.isConnected) return;
      if (active) {
        btn.setAttribute("aria-busy", "true");
        btn.textContent = `Đang xử lý ${completed}/${total}…`;
      } else {
        btn.removeAttribute("aria-busy");
        if (document.body?.dataset?.appStage === "preview") {
          btn.textContent = completed > 0 ? "Tiếp tục xử lý" : "Bắt đầu xử lý";
        }
      }
    });
  }

  async function processingJson(url, options = undefined) {
    const resp = await fetch(url, options);
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    return data;
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function finishSuccessfulProcessing(chapterId) {
    if (chapterId !== currentChapterId) return;
    await refreshChapterManifest(chapterId);
    if (chapterId !== currentChapterId) return;
    if (document.body?.dataset?.appStage === "preview") {
      renderReview();
    }
  }

  async function finishFailedProcessing(chapterId, snapshot, { reconnect = false } = {}) {
    if (chapterId !== currentChapterId) return;
    try {
      await refreshChapterManifest(chapterId);
    } catch (err) {
      console.error("Could not resync chapter after processing failure:", err);
    }
    if (chapterId !== currentChapterId) return;
    setProcessingUi(snapshot);
    if (!reconnect) {
      const first = Array.isArray(snapshot?.errors) ? snapshot.errors[0] : null;
      const detail = first?.message || "Xử lý trang thất bại.";
      showToast(detail, "error");
    }
    if (document.body?.dataset?.appStage === "preview") {
      renderPreview();
    }
  }

  async function monitorProcessingJob(initialSnapshot, chapterId, options = {}) {
    let snapshot = initialSnapshot;
    const reconnect = options.reconnect === true;

    while (
      chapterId === currentChapterId
      && (snapshot?.status === "pending" || snapshot?.status === "running")
    ) {
      setProcessingUi(snapshot);
      await sleep(PROCESS_JOB_POLL_MS);
      if (chapterId !== currentChapterId) return snapshot;
      snapshot = await processingJson(
        `/api/process/jobs/${encodeURIComponent(snapshot.job_id)}`,
      );
    }

    if (chapterId !== currentChapterId) return snapshot;
    setProcessingUi(snapshot);

    if (snapshot?.status === "completed") {
      await finishSuccessfulProcessing(chapterId);
    } else if (snapshot?.status === "failed") {
      await finishFailedProcessing(chapterId, snapshot, { reconnect });
    }
    return snapshot;
  }

  window._processSelectedPagesOnce = async function serverOwnedProcessSelectedPagesOnce() {
    const pages = currentManifest?.pages || [];
    const indices = pages
      .map((page, index) => (page?.skipped ? null : index))
      .filter((index) => index !== null);

    if (!currentChapterId) {
      showToast("Chưa có chương để xử lý.", "error");
      return;
    }
    if (indices.length === 0) {
      showToast("Không có trang nào để xử lý.", "error");
      return;
    }

    const chapterId = currentChapterId;
    processingButtons().forEach((btn) => {
      if (!btn || !btn.isConnected) return;
      btn.setAttribute("aria-busy", "true");
      btn.textContent = "Đang lưu vùng loại trừ…";
    });

    try {
      if (typeof window.flushExcludedRegionSaves === "function") {
        await window.flushExcludedRegionSaves(chapterId);
      }
      if (chapterId !== currentChapterId) return;

      // One request transfers ownership of the *entire* chapter plan to the
      // backend. The browser never submits follow-up batches, so F5/tab close
      // cannot truncate the remaining inpaint queue.
      const snapshot = await processingJson("/api/process/chapter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chapter_id: chapterId,
          page_indices: indices,
          workers: getWorkersSetting(),
        }),
      });
      return await monitorProcessingJob(snapshot, chapterId);
    } catch (err) {
      if (chapterId === currentChapterId) {
        showToast("Xử lý trang thất bại: " + err.message, "error");
      }
    } finally {
      if (chapterId === currentChapterId && document.body?.dataset?.appStage === "preview") {
        const latest = await window.getCurrentProcessingJob(chapterId).catch(() => null);
        if (!latest?.active) setProcessingUi(latest || { status: "idle", completed: 0, total: indices.length });
      }
    }
  };

  window.getCurrentProcessingJob = async function getCurrentProcessingJob(chapterId) {
    if (!chapterId) return null;
    return await processingJson(
      `/api/process/chapter/${encodeURIComponent(chapterId)}`,
    );
  };

  window.reconnectPageProcessingJob = async function reconnectPageProcessingJob(chapterId) {
    if (!chapterId || chapterId !== currentChapterId) return null;
    let snapshot;
    try {
      snapshot = await window.getCurrentProcessingJob(chapterId);
    } catch (err) {
      console.error("Could not query processing job after resume:", err);
      return null;
    }
    if (!snapshot?.active) {
      setProcessingUi(snapshot || { status: "idle", completed: 0, total: 0 });
      return snapshot;
    }
    return await monitorProcessingJob(snapshot, chapterId, { reconnect: true });
  };

  const previousResumeChapter = window.resumeChapter;
  if (typeof previousResumeChapter === "function") {
    window.resumeChapter = async function resumeChapterWithProcessingReconnect(chapterId) {
      await previousResumeChapter(chapterId);
      if (
        chapterId === currentChapterId
        && document.body?.dataset?.appStage === "preview"
      ) {
        await window.reconnectPageProcessingJob(chapterId);
      }
    };
  }
})();
