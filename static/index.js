// alert('Mandelbrot Set');
const canvas = document.querySelector('canvas');
const ctx = canvas.getContext('2d');
const image_data = ctx.createImageData(canvas.width, canvas.height);
debugger;

ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);
