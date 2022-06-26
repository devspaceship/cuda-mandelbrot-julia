// MIT License

// Copyright (c) 2022 Thomas Saint-Gérand

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

const ZOOMING_COEFFICIENT = 0.05;

const canvas = document.querySelector('canvas');
const ctx = canvas.getContext('2d');
const image_data = ctx.createImageData(canvas.width, canvas.height);
let [x_min, x_max, y_min, y_max] = [-2, 1, -1.5, 1.5];

const fetch_mandelbrot = async () => {
  const res = await fetch('/api/mandelbrot', {
    method: 'POST',
    body: JSON.stringify({ x_min, x_max, y_min, y_max }),
  });
  const raw_data = (await res.json()).data;
  const data = raw_data.substring(0, raw_data.length - 1);
  const bin_data = atob(data);
  return bin_data;
};

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

const display_bin_data = (bin_data) => {
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
          255 * Math.pow(iter / max_iter, 1 / 4),
          255 * Math.pow(iter / max_iter, 1 / 4),
          255
        );
      }
    }
  }
  ctx.putImageData(image_data, 0, 0);
};

const main = async () => {
  const bin_data = await fetch_mandelbrot();
  display_bin_data(bin_data);
};

// TODO Wait 70ms without wheel event to prevent multiple zoom events
const zoom_click = async (event) => {
  const scaling_factor_wheel = 1 / (-event.deltaY * ZOOMING_COEFFICIENT);
  const scaling_factor = 1 / 2;
  const i = event.offsetY;
  const j = event.offsetX;
  const center_x = (j / canvas.width) * (x_max - x_min) + x_min;
  const center_y = (i / canvas.height) * (y_max - y_min) + y_min;
  x_min = center_x - (center_x - x_min) * scaling_factor;
  x_max = center_x + (x_max - center_x) * scaling_factor;
  y_min = center_y - (center_y - y_min) * scaling_factor;
  y_max = center_y + (y_max - center_y) * scaling_factor;

  const bin_data = await fetch_mandelbrot();
  display_bin_data(bin_data);
};

// canvas.addEventListener('wheel', zoom);
canvas.addEventListener('click', zoom_click);

main();
