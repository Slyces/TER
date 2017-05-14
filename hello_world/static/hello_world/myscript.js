$(document)
    .ready(function () {
        // Dropdown
        $('.ui.dropdown')
            .dropdown();

        // Sticky menu
        $('.main.menu')
            .visibility({
                type: 'fixed'
            });

        // Scroll up
        // $('#lol').click($('#menu').animatescroll());
    })
;