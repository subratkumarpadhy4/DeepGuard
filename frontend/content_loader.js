// This script runs in the ISOLATED world of the extension.
// Its only job is to inject the 'interceptor.js' into the MAIN world
// so it can access the real 'navigator.mediaDevices' API.

const script = document.createElement('script');
script.src = chrome.runtime.getURL('interceptor.js');
script.onload = function () {
    this.remove(); // Clean up the tag after injection
    console.log("[DeepGuard] Interceptor injected into page context.");
};
(document.head || document.documentElement).appendChild(script);

// Also inject the UI styles
const link = document.createElement('link');
link.rel = 'stylesheet';
// In a real build, we'd inject a CSS file for the overlay
// link.href = chrome.runtime.getURL('styles.css');
// (document.head || document.documentElement).appendChild(link);
