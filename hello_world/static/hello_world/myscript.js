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
        $('.indexes.menu  .item').tab(
            {history: false}
        );

        $('.lstm.menu  .item').tab(
            {history: false}
        );

        $('.plots.menu  .item').tab(
            {history: false}
        );
    })
;