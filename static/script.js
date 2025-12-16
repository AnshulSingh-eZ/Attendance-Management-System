function startAttendance() {
    document.getElementById('attendance-feed').innerHTML =
        '<img src="/video_feed" alt="Live Attendance Feed">';
    document.getElementById('stop-btn').style.display = 'inline-block';
    document.getElementById('attendance-status').textContent =
        'Attendance monitoring started...';
}

function stopAttendance() {
    document.getElementById('attendance-feed').innerHTML =
        '<p class="placeholder-text">Camera feed will appear here during attendance</p>';
    document.getElementById('stop-btn').style.display = 'none';
    document.getElementById('attendance-status').textContent =
        'Attendance monitoring stopped';
    fetch('/stop_attendance').catch(err => console.error(err));
}
document.getElementById('register-form').addEventListener('submit', function (e) {
    e.preventDefault();
    const formData = new FormData(this);

    document.getElementById('register-feed').innerHTML =
        '<img src="/video_feed_register" alt="Registration Camera Feed">';
    document.getElementById('register-status').textContent =
        'Please face the camera for registration...'; 

    fetch('/register', {
        method: 'POST',
        body: formData
    }).catch(error => console.error('Error:', error));
});