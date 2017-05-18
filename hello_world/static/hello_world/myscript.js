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
        $('.abstract.menu  .item').tab(
            {history: false}
        );

        // Plots
        $('#context1  .indexes.menu  .item').tab({
            context: $('#context1')
        });


        $('.plots.menu  .item').tab(
            {history: false}
        );

        $('#context2 .longshort.menu  .item').tab({
            context: $('#context2')
        });
    })
;