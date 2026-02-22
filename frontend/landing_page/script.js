/**
 * DeepGuard Landing Page Interactivity
 * Handles Sidebar Navigation, Theme Toggling, and View Switching
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const menuBtn = document.getElementById('menuBtn');
    const sidebar = document.getElementById('sidebar');
    const closeSidebar = document.getElementById('closeSidebar');
    const sidebarBackdrop = document.getElementById('sidebarBackdrop');
    const themeToggle = document.getElementById('themeToggle');
    const body = document.body;

    // View Switching
    const sidebarLinks = document.querySelectorAll('.sidebar-link');
    const views = document.querySelectorAll('.view-container');



    // Drag & Drop
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const loadingState = document.getElementById('loading-state');
    const resultState = document.getElementById('result-state');

    // --- Sidebar Logic ---
    function openSidebar() {
        sidebar.classList.add('active');
        sidebarBackdrop.style.display = 'flex';
    }

    function closeSidebarMenu() {
        sidebar.classList.remove('active');
        sidebarBackdrop.style.display = 'none';
    }

    if (menuBtn) {
        menuBtn.addEventListener('click', openSidebar);
    }

    if (closeSidebar) {
        closeSidebar.addEventListener('click', closeSidebarMenu);
    }

    if (sidebarBackdrop) {
        sidebarBackdrop.addEventListener('click', closeSidebarMenu);
    }

    // --- Theme Toggling ---
    const savedTheme = localStorage.getItem('deepguard-theme') || 'light';
    if (savedTheme === 'dark') {
        body.classList.add('dark-theme');
    }

    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            body.classList.toggle('dark-theme');
            const newTheme = body.classList.contains('dark-theme') ? 'dark' : 'light';
            localStorage.setItem('deepguard-theme', newTheme);
        });
    }

    // --- View Switching ---
    sidebarLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();

            // Remove active class from all links
            sidebarLinks.forEach(l => l.classList.remove('active'));
            // Add active class to clicked link
            link.classList.add('active');

            const targetViewId = link.getAttribute('data-view');

            // Hide all views
            views.forEach(view => {
                view.classList.remove('active');
            });

            // Show target view
            const targetView = document.getElementById(targetViewId + 'View');
            if (targetView) {
                targetView.classList.add('active');
            }

            // Close sidebar on mobile/desktop after selection
            closeSidebarMenu();
        });
    });



    // --- Drag & Drop Upload Logic ---
    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.background = 'var(--accent-glow)';
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.style.background = '';
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.background = '';

            if (e.dataTransfer.files.length > 0) {
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });
    }

    function handleFileUpload(file) {
        dropZone.style.display = 'none';
        if (loadingState) loadingState.style.display = 'block';
        if (resultState) resultState.style.display = 'none';

        // Simulate analysis delay
        setTimeout(() => {
            if (loadingState) loadingState.style.display = 'none';
            if (resultState) {
                resultState.style.display = 'block';
                resultState.innerHTML = `
                    <div style="padding: 20px; background: var(--bg); border-radius: 12px; border: 1px solid var(--card-border);">
                        <h3 style="color: var(--accent); margin-bottom: 10px;">Analysis Complete</h3>
                        <p style="font-size: 0.9rem; margin-bottom: 5px;"><strong>File:</strong> ${file.name}</p>
                        <p style="font-size: 0.9rem; color: var(--text-soft);"><strong>Status:</strong> No synthesis artifacts detected. Authentic source.</p>
                        <button class="btn-secondary" style="margin-top: 15px;" onclick="resetUpload()">Analyze Another File</button>
                    </div>
                `;
            }
        }, 2000);
    }

    // Global function to reset upload
    window.resetUpload = function () {
        if (dropZone) dropZone.style.display = 'flex';
        if (resultState) resultState.style.display = 'none';
        if (fileInput) fileInput.value = '';
    };
});
