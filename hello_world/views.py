# Django
from django.views import generic
from hello_world.models import HistoricIndexes
from django.shortcuts import render

# Bokeh
from bokeh.plotting import figure
from bokeh.embed import components
from hello_world.Scripts import plotting

def MetaChapter(file):
    class Chapter(generic.base.TemplateView):
        template_name = 'hello_world/' + file + '.html'
    return Chapter

class IndexView(generic.base.TemplateView):
    template_name = 'hello_world/index.html'
    context_object_name = 'context'

    def get_context_data(self, *args, **kwargs):
        context = super(IndexView, self).get_context_data(**kwargs)

        plots = plotting.plot_feedforward('1')

        context['script_bokeh'], context['divs'] = components(plots)

        return context

concepts = MetaChapter('concepts')