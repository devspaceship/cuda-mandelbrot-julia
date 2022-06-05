// alert('Mandelbrot Set');
const canvas = document.querySelector('canvas');
const ctx = canvas.getContext('2d');
const image_data = ctx.createImageData(canvas.width, canvas.height);

const main = async () => {
  const res = await fetch('/api/mandelbrot');
  const data = (await res.json()).data;
  debugger;
};

main();

// TODO Make request to API
// TODO Make image data
// TODO Draw image data
