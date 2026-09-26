const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const routes = { '/': ['index.html', 'text/html'], '/client.js': ['client.js', 'text/javascript'], '/ethers.js': ['ethers.js', 'text/javascript'], '/artifact.json': ['artifact.json', 'application/json'], '/compiler-input.json': ['compiler-input.json', 'application/json'], '/HavnNodeRewardClaims.sol': ['HavnNodeRewardClaims.sol', 'text/plain'] };
http.createServer((req, res) => {
  if (req.headers.host !== '127.0.0.1:8787' || req.method !== 'GET' || !routes[req.url]) { res.writeHead(404); res.end(); return; }
  const [file, type] = routes[req.url];
  res.writeHead(200, { 'Content-Type': type + '; charset=utf-8', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', 'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'" });
  fs.createReadStream(path.join(__dirname, file)).pipe(res);
}).listen(8787, '127.0.0.1', () => console.log('Rewards deployment: http://127.0.0.1:8787'));
