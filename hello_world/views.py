# from django.shortcuts import get_object_or_404, render
# from django.http import HttpResponseRedirect
# from django.urls import reverse
import os

# Django
from django.views import generic
from hello_world.models import HistoricIndexes
from django.shortcuts import render

# Bokeh
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import components
from hello_world.Scripts import Visualize


def simple_chart(request):
    plot = figure()
    plot.circle([1, 2], [3, 4])

    script, div = components(plot, CDN)

    print(script, div)

    return render(request, "hello_world/simple_chart.html",
                  {"script": script, "div": div})


class IndexView(generic.base.TemplateView):
    template_name = 'hello_world/index.html'
    context_object_name = 'context'

    def get_context_data(self, *args, **kwargs):
        context = super(IndexView, self).get_context_data(**kwargs)

        # indexes = 'Nasdaq DowJones SnP500 Rates'.split()
        # plots = Visualize.plot_feedforward(path='hello_world/Scripts/')
        #
        # for index in indexes:
        #     script, div = components(plots[index], CDN)
        #     context[index + '_script'] = script
        #     context[index + '_div'] = div

        return context
