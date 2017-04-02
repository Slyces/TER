# Django
from django.views import generic
from hello_world.models import HistoricIndexes
from django.shortcuts import render

# Bokeh
from bokeh.plotting import figure
from bokeh.embed import components
from hello_world.Scripts import plotting


def simple_chart(request):
    plot = figure()
    plot.circle([1, 2], [3, 4])

    script, div = components(plot)

    print(script, div)

    return render(request, "hello_world/simple_chart.html",
                  {"script": script, "div": div})


class IndexView(generic.base.TemplateView):
    template_name = 'hello_world/index.html'
    context_object_name = 'context'

    def get_context_data(self, *args, **kwargs):
        context = super(IndexView, self).get_context_data(**kwargs)

        plots = plotting.plot_feedforward('1')

        script, divs = components(plots)
        context['script_bokeh'] = script

        for index in divs:
            context[index + '_div'] = divs[index]

        return context
