(function () {

function initCarousels() {
  document.querySelectorAll(".image-carousel").forEach(carousel => {
    if (carousel.dataset.ready) return;
    carousel.dataset.ready = "true";

    const images = [...carousel.querySelectorAll("img")];

    if (images.length <= 1) return;

    let current = 0;

    images[0].classList.add("active");

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

    images.forEach((_, i) => {
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
        images.forEach((img, i) => img.classList.toggle("active", i === current));
        dotEls.forEach((dot, i) => dot.classList.toggle("active", i === current));
    }

    prev.onclick = () => {
        current = (current - 1 + images.length) % images.length;
        update();
    };

    next.onclick = () => {
      current = (current + 1) % images.length;
      update();
    };
  });
}

if (window.document$) {
  document$.subscribe(initCarousels);
}
else {
  document.addEventListener("DOMContentLoaded", initCarousels);
}

})();