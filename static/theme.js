const THEME_KEY = "sp-analytics-theme";

const THEME_LABELS = {
    "spotify-dark": "Spotify Dark",
    "spotify-light": "Spotify Light",
    "midnight": "Midnight",
    "forest": "Forest",
    "sunflower": "Sunflower Sunrise",
    "marshmellow-pink": "Marshmellow Pink"
};

function applyTheme(theme) {

    document.documentElement.setAttribute(
        "data-theme",
        theme
    );

    localStorage.setItem(
        THEME_KEY,
        theme
    );

    const themeLabel =
        document.getElementById("themeLabel");

    if (themeLabel) {

        themeLabel.textContent =
            THEME_LABELS[theme] || theme;

    }

    document
        .querySelectorAll(".sp-theme-option")
        .forEach(option => {

            option.classList.toggle(
                "selected",
                option.dataset.theme === theme
            );

        });
}


document.addEventListener(
    "DOMContentLoaded",
    () => {

        applyTheme(
            localStorage.getItem(THEME_KEY)
            || "spotify-dark"
        );


        const themeBtn =
            document.getElementById("themeBtn");

        const themeMenu =
            document.getElementById("themeMenu");


        if (themeBtn && themeMenu) {

            themeBtn.addEventListener(
                "click",
                event => {

                    event.stopPropagation();

                    themeMenu.classList.toggle(
                        "open"
                    );

                }
            );


            document
                .querySelectorAll(".sp-theme-option")
                .forEach(option => {

                    option.addEventListener(
                        "click",
                        () => {

                            applyTheme(
                                option.dataset.theme
                            );

                            themeMenu.classList.remove(
                                "open"
                            );

                        }
                    );

                });


            document.addEventListener(
                "click",
                () => {

                    themeMenu.classList.remove(
                        "open"
                    );

                }
            );

        }

    }
);