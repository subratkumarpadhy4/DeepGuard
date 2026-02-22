/**
 * DeepGuard Extension Popup Logic
 * CSP-compliant (no inline scripts)
 */

// Immediate theme application to prevent flash
(function() {
    const savedTheme = localStorage.getItem('deepguard-theme') || 'light';
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
    }
})();

document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('themeToggle');
    const logoLink = document.getElementById('logoLink');
    const body = document.body;

    // --- Theme Toggle Logic ---
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            body.classList.toggle('dark-theme');
            const newTheme = body.classList.contains('dark-theme') ? 'dark' : 'light';
            localStorage.setItem('deepguard-theme', newTheme);
            console.log("Theme switched to:", newTheme);
        });
    }

    // --- Logo Link (Navigate to Landing Page) ---
    if (logoLink) {
        logoLink.addEventListener('click', () => {
            if (typeof chrome !== 'undefined' && chrome.tabs) {
                const url = chrome.runtime.getURL("landing_page/index.html");
                chrome.tabs.create({ url: url });
            } else {
                window.open('landing_page/index.html', '_blank');
            }
        });
    }

    // --- Initialize other UI elements if needed ---
    console.log("DeepGuard Popup Initialized");
});
