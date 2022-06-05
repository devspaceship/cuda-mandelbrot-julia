// alert('Mandelbrot Set');
const canvas = document.querySelector('canvas');
const ctx = canvas.getContext('2d');
const image_data = ctx.createImageData(canvas.width, canvas.height);

const main = async () => {
  const res = await fetch('/api/mandelbrot');
  const raw_data = (await res.json()).data;
  const data = raw_data.substring(0, raw_data.length - 1);
  const bin_data = atob(data);
  for (let i = 0; i < canvas.height; i++) {
    for (let j = 0; j < canvas.width; j++) {
      const index = i * canvas.width + j;
      const iter = bin_data.charCodeAt(index);
      image_data.data[index * 4] = 255 - iter;
      image_data.data[index * 4 + 1] = 255 - iter;
      image_data.data[index * 4 + 2] = 255 - iter;
      image_data.data[index * 4 + 3] = 255;
    }
  }
  ctx.putImageData(image_data, 0, 0);
};

main();

// TODO Make image data
// TODO Draw image data
