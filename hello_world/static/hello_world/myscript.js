/**
 * Created by Slyces on 5/17/17.
 */
$(document)
    .ready(function () {

        // Sticky menu
        $('.main.menu')
            .visibility({
                type: 'fixed'
            });

        // Abstract
        $('.abstract.menu .item').
            tab(
                {history:false}
        );

    //     $('.dropdown')
    //         .dropdown();
    })
;