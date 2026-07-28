self.addEventListener('install', (e) => {
  e.waitUntil(caches.open('plant-cache').then((cache) => cache.addAll(['./', './index.html'])));
});
self.addEventListener('fetch', (e) => {
  e.respondWith(caches.match(e.request).then((res) => res || fetch(e.request)));
});
