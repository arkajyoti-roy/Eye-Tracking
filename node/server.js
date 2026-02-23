const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');

const app = express();
app.use(cors());

const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*" } });

let frameCount = 0;

io.on('connection', (socket) => {
    console.log(`🔌 New connection established: ${socket.id}`);

    socket.on('python_data', (data) => {
        io.emit('pupil_move', data);
        
        frameCount++;
        if (frameCount % 10 === 0) {
            // Updated to safely log the new metrics!
            console.log(`Live Data -> Attn: ${data.metrics.attention}% | Blinks: ${data.metrics.blinks} | Head: ${data.metrics.head}`);
        }
    });

    socket.on('disconnect', () => {
        console.log(`❌ Disconnected: ${socket.id}`);
    });
});

const PORT = 5000;
server.listen(PORT, () => {
    console.log(`🚀 Node Master Server running on http://localhost:${PORT}`);
});