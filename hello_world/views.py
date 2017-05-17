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

class ApplicationsView(generic.base.TemplateView):
    template_name = 'hello_world/application.html'
    context_object_name = 'context'

    def get_context_data(self, *args, **kwargs):
        context = super(ApplicationsView, self).get_context_data(**kwargs)

        plots = plotting.plot_feedforward('1')
        plots['lstm'] = plotting.plot_lmst()

        context['script'], context['divs'] = components(plots)
        return context

introduction = MetaChapter('introduction')
concepts = MetaChapter('concepts')
finance = MetaChapter('finance')
application = MetaChapter('application')
conclusion = MetaChapter('conclusion')
test = MetaChapter('test')
index = MetaChapter('index')
