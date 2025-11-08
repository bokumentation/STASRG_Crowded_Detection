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

async function fetchCountData() {
    try {
        const response = await fetch("/count_data");
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        const data = await response.json();

        document.getElementById("entry-count").textContent = data.entry_count;
        document.getElementById("exit-count").textContent = data.exit_count;
        document.getElementById("current-count").textContent = data.current_count;

        const currentCount = data.current_count;
        const statusElement = document.getElementById("status");

        // Perbarui status
        if (currentCount > 150) {
        statusElement.textContent = "Status: Overcrowded";
        statusElement.classList.remove("text-green-600");
        statusElement.classList.add("text-red-600");
        } else {
        statusElement.textContent = "Status: Normal";
        statusElement.classList.remove("text-red-600");
        statusElement.classList.add("text-green-600");
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
    fetchCountData();
    setInterval(fetchCountData, 1000); // Perbarui data setiap 1 detik
    setInterval(gantiGambar, 3000); // Mengganti gambar setiap 3 detik
});
