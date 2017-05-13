$(document)
    .ready(function () {

        // fix segment to page on passing
        $('.header.sticky')
            .sticky({
                context: '#chapter1'
            })
        ;

    })
;