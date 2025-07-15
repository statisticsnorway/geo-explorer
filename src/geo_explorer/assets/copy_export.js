window.addEventListener('DOMContentLoaded', function() {
    var btn = document.getElementById("copy-export");
    if (btn) {
        btn.addEventListener("click", function() {
            var exportText = document.getElementById("export-text");
            if (exportText) {
                // Get the visible text, including line breaks
                var text = exportText.innerText || exportText.textContent;
                navigator.clipboard.writeText(text).then(function() {
                    btn.innerText = "Copied!";
                    setTimeout(function() {
                        btn.innerText = "Copy code";
                    }, 1500);
                });
            }
        });
    }
});