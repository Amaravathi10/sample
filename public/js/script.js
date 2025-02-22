window.addEventListener('scroll', () => {
    document.querySelectorAll('.bloom-section').forEach(section => {
        if (section.getBoundingClientRect().top < window.innerHeight - 100) {
            section.classList.add('visible');
        }
    });
});