""" Plugin for OpenAleaLab """


class RootEditorWidgetPlugin(object):

    """ applet plugin for RootEditor """
    name = 'RootEditor'
    alias = 'RootEditor'

    def __call__(self):
        from rhizoscan.root.editor import RootEditorWidget
        return RootEditorWidget

    def graft(self, **kwds):
        mainwindow = kwds['oa_mainwin'] if 'oa_mainwin' in kwds else None
        applet = kwds['applet'] if 'applet' in kwds else None
        if applet is None or mainwindow is None:
            return

        # widget
        RootEditorWidget = self()
        self._applet = RootEditorWidget()
        mainwindow.add_applet(self._applet, self.alias, area='outputs')

        # actions
        actions = self._applet.get_plugin_actions()
        if actions:
            for action in actions:
                # Add actions in PanedMenu
                mainwindow.menu.addBtnByAction('RootEditor', *action)

                # add action in classical menu
                group_name, act, btn_type = action
                mainwindow.add_action_to_existing_menu(action=act, menu_name='RootEditor', sub_menu_name=group_name)

    def instance(self):
        # Write your code here
        pass


class SeedMapWidgetPlugin(object):

    """ applet plugin for RootEditor """
    name = 'SeedMapEditor'
    alias = 'SeedMapEditor'

    def __call__(self):
        from rhizoscan.root.image.gui.seed.seed_editor import SeedMapWidget
        return SeedMapWidget

    def graft(self, **kwds):
        mainwindow = kwds['oa_mainwin'] if 'oa_mainwin' in kwds else None
        applet = kwds['applet'] if 'applet' in kwds else None
        if applet is None or mainwindow is None:
            return

        SeedMapWidget = self()
        self._applet = SeedMapWidget()
        mainwindow.add_applet(self._applet, self.alias, area='outputs')

        # actions
        ##actions = self._applet.get_plugin_actions()
        ##if actions:
        ##    for action in actions:
        ##        # Add actions in PanedMenu
        ##        ##mainwindow.menu.addBtnByAction('RootEditor', *action)
        ##
        ##        # add action in classical menu
        ##        group_name, act, btn_type = action
        ##        mainwindow.add_action_to_existing_menu(action=act, menu_name='RootEditor', sub_menu_name=group_name)

    def instance(self):
        # Write your code here
        pass


from openalea.oalab.plugins.labs.default import EmptyLab


class RhizoScanLab(EmptyLab):
    name = 'rhizoscan'
    alias = 'Rhizoscan'
    applet_names = [
        'RootEditor',
        'SeedMapEditor',
        'ProjectManager2',
        'ControlManager',
        'PkgManagerWidget',
        'EditorManager',
        'Logger',
        'HelpWidget',
        'HistoryWidget',
        'World',
        'Plot2d',
    ]

    layout = {
        "children": {"0": [1, 2], "1": [5, 6]},
        "parents": {
            "0": None,
            "1": 0,
            "2": 0,
            "5": 1,
            "6": 1
        },
        "properties": {
            "0": {
                "amount": 0.7214854111405835,
                "splitDirection": 2
            },
            "1": {
                "amount": 0.10309278350515463,
                "splitDirection": 2
            },
            "2": {"widget": {"applets": [{"name": "ShellWidget", }], }},
            "5": {"widget": {"applets": [{"name": "ContextualMenu", }], }},
            "6": {"widget": {"applets": [{"name": "RootEditor", }], }}
        }
    }

    def __call__(self, mainwin=None):
        if mainwin is None:
            return self.__class__
        from openalea.vpltk.plugin import iter_plugins
        session = mainwin.session

        # 1. Load applet
        # 2. Place applet following given order,
        plugins = {}
        for plugin in iter_plugins('oalab.applet', debug=session.debug_plugins):
            if plugin.name in self.applet_names:
                plugin = plugin()
                plugins[plugin.name] = plugin
                mainwin.add_plugin(plugin)

        # 3. Once the applet is loaded, init them
        mainwin.initialize()
