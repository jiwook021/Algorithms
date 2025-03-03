// ws-signaling.js
// CommonJS version—no need for “type”: “module”

const WebSocket = require('ws');

const wss = new WebSocket.Server({ port: 8888 });
let peers = [];

wss.on('connection', ws => {
  peers.push(ws);

  ws.on('message', msg => {
    // Relay to everyone else
    for (const p of peers) {
      if (p !== ws && p.readyState === WebSocket.OPEN) {
        p.send(msg);
      }
    }
  });

  ws.on('close', () => {
    peers = peers.filter(p => p !== ws);
  });
});

console.log('Signaling server running on ws://localhost:8888');
