
    // Show the floating div in the center for 5 seconds
    setTimeout(() => {
        const floatingDiv = document.getElementById("floating-div");
        floatingDiv.style.position = "fixed";
        floatingDiv.style.top = "auto";
        floatingDiv.style.left = "auto";
        floatingDiv.style.bottom = "20px"; // Position from the bottom
        floatingDiv.style.right = "20px"; // Position from the right
        floatingDiv.style.transform = "none";
    }, 1000);

    // Toggle instructions visibility
    document.getElementById("your-goal-header").onclick = () => {
        const content = document.getElementById("your-goal-content");
        content.style.display = content.style.display === "none" ? "block" : "none";
    };