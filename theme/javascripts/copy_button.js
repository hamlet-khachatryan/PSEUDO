document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('div.highlight').forEach(function (block) {
        var button = document.createElement('button');
        button.className = 'copy-btn';
        button.textContent = 'Copy';
        block.appendChild(button);

        button.addEventListener('click', function () {
            var code = block.querySelector('pre').textContent;

            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(code).then(function () {
                    button.textContent = 'Copied!';
                    setTimeout(function () { button.textContent = 'Copy'; }, 2000);
                });
            } else {
                // Fallback for older browsers
                var ta = document.createElement('textarea');
                ta.value = code;
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
                button.textContent = 'Copied!';
                setTimeout(function () { button.textContent = 'Copy'; }, 2000);
            }
        });
    });
});