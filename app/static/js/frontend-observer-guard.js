(() => {
  "use strict";

  const NativeMutationObserver = window.MutationObserver;
  if (typeof NativeMutationObserver !== "function" || NativeMutationObserver._mtPageViewGuard) {
    return;
  }

  function GuardedMutationObserver(callback) {
    const observer = new NativeMutationObserver(callback);
    const nativeObserve = observer.observe.bind(observer);

    observer.observe = (target, options = {}) => {
      if (target?.id !== "page-view") {
        return nativeObserve(target, options);
      }

      // The stage renderers replace direct children of #page-view. Watching the
      // entire subtree makes progress-label/text changes wake expensive layout
      // and polish observers during /api/process_pages. Keep the structural
      // fallback while ignoring deep UI churn.
      const rootOnly = {
        ...options,
        childList: true,
        subtree: false,
        attributes: false,
        characterData: false,
      };
      delete rootOnly.attributeFilter;
      return nativeObserve(target, rootOnly);
    };

    return observer;
  }

  GuardedMutationObserver.prototype = NativeMutationObserver.prototype;
  Object.setPrototypeOf(GuardedMutationObserver, NativeMutationObserver);
  GuardedMutationObserver._mtPageViewGuard = true;
  GuardedMutationObserver._mtNative = NativeMutationObserver;
  window.MutationObserver = GuardedMutationObserver;
})();
