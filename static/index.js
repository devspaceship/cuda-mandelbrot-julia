// alert('Mandelbrot Set');
const canvas = document.querySelector('canvas');
const ctx = canvas.getContext('2d');
const image_data = ctx.createImageData(canvas.width, canvas.height);

const get_max_iter = (bin_data) => {
  let max_iter = 0;
  for (let i = 0; i < canvas.height; i++) {
    for (let j = 0; j < canvas.width; j++) {
      const index = i * canvas.width + j;
      const iter = bin_data.charCodeAt(index);
      if (iter > max_iter && iter !== 255) {
        max_iter = iter;
      }
    }
  }
  return max_iter;
};

const set_pixel = (i, j, r, g, b, a) => {
  const index = i * canvas.width + j;
  image_data.data[index * 4] = r;
  image_data.data[index * 4 + 1] = g;
  image_data.data[index * 4 + 2] = b;
  image_data.data[index * 4 + 3] = a;
};

const main = async () => {
  const res = await fetch('/api/mandelbrot');
  const raw_data = (await res.json()).data;
  const data = raw_data.substring(0, raw_data.length - 1);
  const bin_data = atob(data);
  const max_iter = get_max_iter(bin_data);
  for (let i = 0; i < canvas.height; i++) {
    for (let j = 0; j < canvas.width; j++) {
      const index = i * canvas.width + j;
      const iter = bin_data.charCodeAt(index);
      if (iter === 255) {
        set_pixel(i, j, 0, 0, 0, 255);
      } else {
        set_pixel(
          i,
          j,
          0,
          255 * Math.pow(iter / max_iter, 1 / 3),
          255 * Math.pow(iter / max_iter, 1 / 3),
          255
        );
      }
    }
  }
  ctx.putImageData(image_data, 0, 0);
};

main();
