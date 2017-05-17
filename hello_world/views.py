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

        indexes = "Rates DowJones Nasdaq SnP500".split()
        context['indexes'] = indexes

        ffp, ffe, ffn = plotting.plot_feedforward('1')

        lstmp, lstme = plotting.plot_lmst()

        ffp_script, ffp_divs = components(ffp)
        ffe_script, ffe_divs = components(ffe)
        ffn_script, ffn_divs = components(ffn)

        lstmp_script, lstmp_divs = components(lstmp)
        lstme_script, lstme_divs = components(lstme)

        context['plot'] = {
            'lstm': lstmp_divs
        }
        context['error'] = {
            'lstm': lstme_divs
        }
        context['normalised'] = {}
        context['script'] = '\n'.join([lstmp_script, lstme_script])
        for index in indexes:
            context['plot'][index] = ffp_divs[index]
            context['error'][index] = ffe_divs[index]
            context['normalised'][index] = ffn_divs[index]
        context['script'] += '\n' + '\n'.join([ffp_script, ffe_script, ffn_script])
        return context


introduction = MetaChapter('introduction')
concepts = MetaChapter('concepts')
finance = MetaChapter('finance')
application = MetaChapter('application')
conclusion = MetaChapter('conclusion')
test = MetaChapter('test')
index = MetaChapter('index')
