const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');

const app = express();
app.use(cors());

// Create the HTTP server
const server = http.createServer(app);

// Setup the Master Socket.io Server
const io = new Server(server, {
    cors: { origin: "*" }
});

let frameCount = 0;

io.on('connection', (socket) => {
    console.log(`🔌 New connection established: ${socket.id}`);

    // Listen for the live data coming from Python
    socket.on('python_data', (data) => {
        // Broadcast it to the HTML/React frontend
        io.emit('pupil_move', data);
        
        // Log to terminal every 10 frames
        frameCount++;
        if (frameCount % 10 === 0) {
            console.log(`Live Data -> Left X: ${data.left.x.toFixed(3)} | Right X: ${data.right.x.toFixed(3)}`);
        }
    });

    socket.on('disconnect', () => {
        console.log(`❌ Disconnected: ${socket.id}`);
    });
});

// Start the Node server
const PORT = 5000;
server.listen(PORT, () => {
    console.log(`🚀 Node Master Server running on http://localhost:${PORT}`);
});