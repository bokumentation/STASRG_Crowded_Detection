let crowdThreshold = 150;
let currentCountChart;
let maxDataPoints = 50; // Maksimal 50 titik data terakhir
let timeLabels = [];
let countData = [];
let lastChartUpdate = Date.now(); // Waktu terakhir update grafik
const gambarArray = [
    "/static/stas.png",
    "/static/telkom_logo.png",
    "/static/econique_logo.png"
];
let index_gambar = 0; // Indeks gambar saat ini

// NEW: Function to fetch the crowd threshold from the backend
async function fetchConfig() {
    try {
        const response = await fetch("/api/config");
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        const config = await response.json();
        
        // Update the global threshold variable
        if (config.capacity && config.capacity.max_crowd_count) {
            crowdThreshold = config.capacity.max_crowd_count;
            console.log(`Crowd threshold set to: ${crowdThreshold}`);
        }
    } catch (error) {
        console.error("Error fetching config data, using default threshold (150).", error);
    }
}

async function fetchCountData() {
    try {
        const response = await fetch("/count_data");
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        const data = await response.json();

        document.getElementById("entry-count").textContent = data.entry_count;
        document.getElementById("exit-count").textContent = data.exit_count;
        document.getElementById("current-count").textContent = data.current_count;

        const currentCount = data.current_count;
        const statusTextElement = document.getElementById("status");
        const statusCardElement = document.getElementById("status-card");

        // Ensure all background classes are removed to keep the card white
        statusCardElement.classList.remove("bg-red-600", "bg-green-600");
        statusCardElement.classList.add("bg-white"); // Explicitly set to white

        // Determine Overcrowded (Assuming your threshold is still 150 or fetched)
        if (currentCount > crowdThreshold) { 
            statusTextElement.textContent = "PENUH";

            statusTextElement.textContent = "PENUH";

            // Apply Red Text Color (text-red-600)
            statusTextElement.classList.remove("text-gray-800", "text-green-600");
            statusTextElement.classList.add("text-red-600"); // Red text for warning

            // // --- RED/PENUH STATE ---
            // statusCardElement.classList.remove("bg-green-600", "bg-white");
            // statusCardElement.classList.add("bg-red-600");
            
            // // Text color is white for contrast on the red background
            // statusTextElement.classList.remove("text-gray-800", "text-green-600"); 
            // statusTextElement.classList.add("text-white");

        } else {
            statusTextElement.textContent = "Normal";
            
            // Apply Green/Dark Text Color for Normal state
            statusTextElement.classList.remove("text-red-600", "text-green-600"); 
            statusTextElement.classList.add("text-gray-800"); // Default dark gray text (Normal)
            
            // --- GREEN/NORMAL STATE (FIXED LOGIC) ---
            // // 1. Remove the opposing color
            // statusCardElement.classList.remove("bg-red-600", "bg-white"); 
            // statusCardElement.classList.add("bg-green-600"); 
            
            // // 3. Set text color to white for contrast on the green background
            // statusTextElement.classList.remove("text-gray-800", "text-red-600"); 
            // statusTextElement.classList.add("text-white"); 
        }
    } catch (error) {
        console.error("Error fetching count data:", error);
    }
}


async function resetCount() {
    try {
        const response = await fetch("/reset_count", { method: "POST" });
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        // Set ulang nilai di UI
        document.getElementById("entry-count").textContent = 0;
        document.getElementById("exit-count").textContent = 0;
        document.getElementById("current-count").textContent = 0;

    } catch (error) {
        console.error("Error resetting count:", error);
    }
}


// Fungsi untuk mengganti gambar
function gantiGambar() {
    index_gambar = (index_gambar + 1) % gambarArray.length; // Menghitung indeks gambar berikutnya
    document.getElementById('logo').src = gambarArray[index_gambar]; // Mengganti src gambar
}


window.addEventListener("load", () => {
    // 1. Fetch config first
    fetchConfig();

    // 2. Start data updateds
    fetchCountData();
    setInterval(fetchCountData, 1000); // Perbarui data setiap 1 detik
    setInterval(gantiGambar, 3000); // Mengganti gambar setiap 3 detik
});
