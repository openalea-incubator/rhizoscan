""" Plugin for OpenAleaLab """
        
        
class RootEditorWidgetPlugin(object):
    """ applet plugin for RootEditor """
    name = 'RootEditor'
    alias = 'RootEditor'

    def __call__(self, mainwindow):
        """ Create widget """
        
        # widget
        import treeeditor
        from rhizoscan.root.editor import RootEditorWidget
        self._applet = RootEditorWidget(theme=treeeditor.OATHEME)
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

    def __call__(self, mainwindow):
        """ Create widget """
        
        # widget
        from rhizoscan.root.image.gui.seed.seed_editor import SeedMapWidget
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

class RootEditorLabPlugin(object):
    name = 'rooteditor'
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
        'Viewer3D',
        'World',
        'Plot2d',
        ]


    def __call__(self, mainwin):
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
