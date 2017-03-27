# from django.shortcuts import get_object_or_404, render
# from django.http import HttpResponseRedirect
# from django.urls import reverse
from django.views import generic
from hello_world.models import Messages


from django.shortcuts import render
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import components

def simple_chart(request):
    plot = figure()
    plot.circle([1,2], [3,4])

    script, div = components(plot, CDN)

    return render(request, "simple_chart.html", {"the_script": script, "the_div": div})

class IndexView(generic.ListView):
    template_name = 'hello_world/index.html'

    def get_queryset(self, *args, **kwargs):
        pass
