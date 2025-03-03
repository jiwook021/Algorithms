const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const app = express();
const port = 8080;

// Configure storage for uploaded files
const storage = multer.diskStorage({
    // Set destination folder for uploads
    destination: (req, file, cb) => {
        const uploadDir = path.join(__dirname, 'uploads');
        // Create the uploads directory if it doesn't exist
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir, { recursive: true });
        }
        cb(null, uploadDir);
    },
    // Set filename for uploaded files (timestamp + original name)
    filename: (req, file, cb) => {
        const uniqueName = `${Date.now()}-${file.originalname}`;
        cb(null, uniqueName);
    }
});

// Create multer upload middleware with file size limit
const upload = multer({ 
    storage: storage,
    limits: {
        fileSize: 1024 * 1024 * 500 // 500 MB limit
    }
});

// Serve static files from the current directory
app.use(express.static(__dirname));

// Serve uploaded videos from the uploads directory
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Serve the HTML page on root URL
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Handle video uploads
app.post('/upload', upload.single('video'), (req, res) => {
    // Check if a file was uploaded
    if (!req.file) {
        return res.status(400).json({ error: 'No video file uploaded' });
    }
    
    // Return the URL to the uploaded video
    const videoUrl = `/uploads/${req.file.filename}`;
    res.status(200).json({
        message: 'Video uploaded successfully',
        videoUrl: videoUrl,
        fileName: req.file.originalname,
        size: req.file.size
    });
});

// API endpoint to get a list of uploaded videos
app.get('/videos', (req, res) => {
    const uploadDir = path.join(__dirname, 'uploads');
    
    // Create the directory if it doesn't exist
    if (!fs.existsSync(uploadDir)) {
        fs.mkdirSync(uploadDir, { recursive: true });
        return res.json({ videos: [] });
    }
    
    // Read the directory contents
    fs.readdir(uploadDir, (err, files) => {
        if (err) {
            return res.status(500).json({ error: 'Failed to read uploads directory' });
        }
        
        // Filter for video files
        const videoFiles = files.filter(file => {
            const ext = path.extname(file).toLowerCase();
            return ['.mp4', '.webm', '.mov', '.avi', '.mkv'].includes(ext);
        });
        
        // Create a list of video objects with their URLs
        const videos = videoFiles.map(file => {
            return {
                name: file,
                url: `/uploads/${file}`
            };
        });
        
        res.json({ videos });
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Video upload server running at http://localhost:${port}`);
});