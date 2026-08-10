(function () {

function initCarousels() {
  document.querySelectorAll(".image-carousel").forEach(carousel => {
    if (carousel.dataset.ready) return;
    carousel.dataset.ready = "true";

    const slides = [...carousel.querySelectorAll("img, video")];

    if (slides.length <= 1) return;

    let current = 0;

    slides[0].classList.add("active");

    // caption

    const caption = document.createElement("div");
    caption.className = "carousel-caption";

    carousel.after(caption);

    function getCaption(slide) {
      if (slide.tagName === "IMG") {
        return slide.alt || "";
      }

      if (slide.tagName === "VIDEO") {
        return slide.dataset.caption || "";
      }

      return "";
    }

    // buttons

    const prev = document.createElement("button");
    prev.className = "prev";
    prev.innerHTML = "❮";

    const next = document.createElement("button");
    next.className = "next";
    next.innerHTML = "❯";

    carousel.appendChild(prev);
    carousel.appendChild(next);

    // dots

    const dots = document.createElement("div");
    dots.className = "carousel-dots";

    const dotEls = [];

    slides.forEach((_, i) => {
      const dot = document.createElement("div");
      dot.className = "carousel-dot";

      if (i === 0)
        dot.classList.add("active");

      dot.onclick = () => {
        current = i;
        update();
      };

      dots.appendChild(dot);
      dotEls.push(dot);
    });

    carousel.after(dots);

    function update() {
      slides.forEach((slide, i) => { slide.classList.toggle("active", i === current); });

      dotEls.forEach((dot, i) => { dot.classList.toggle("active", i === current); });

      // Pause videos when switching away from them
      slides.forEach((slide, i) => {
        if (slide.tagName === "VIDEO" && i !== current) {
          slide.pause();
        }
      });

      caption.textContent = getCaption(slides[current]);
    }

    prev.onclick = () => {
      current = (current - 1 + slides.length) % slides.length;
      update();
    };

    next.onclick = () => {
      current = (current + 1) % slides.length;
      update();
    };

    // Initialize caption
    update();
  });
}

if (window.document$) {
  document$.subscribe(initCarousels);
}
else {
  document.addEventListener("DOMContentLoaded", initCarousels);
}

})();