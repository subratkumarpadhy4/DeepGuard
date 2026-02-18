document.addEventListener('DOMContentLoaded', () => {
    const logoLink = document.getElementById('logoLink');
    if (logoLink) {
        logoLink.addEventListener('click', () => {
            // Open the landing page in a new tab
            const url = chrome.runtime.getURL("landing_page/index.html");
            chrome.tabs.create({ url: url });
        });
    }

    // Initialize toggle states if any are saved in storage (optional enhancement)
    // currently we just handle the logo click
});
