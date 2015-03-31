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

    layout = \
{
  "children": {
    "0": [
      1, 
      2
    ], 
    "2": [
      3, 
      4
    ], 
    "3": [
      5, 
      6
    ], 
    "4": [
      7, 
      8
    ], 
    "7": [
      11, 
      12
    ], 
    "8": [
      9, 
      10
    ]
  }, 
  "interface": "ISplittableUi", 
  "parents": {
    "0": None, 
    "1": 0, 
    "2": 0, 
    "3": 2, 
    "4": 2, 
    "5": 3, 
    "6": 3, 
    "7": 4, 
    "8": 4, 
    "9": 8, 
    "10": 8, 
    "11": 7, 
    "12": 7
  }, 
  "properties": {
    "0": {
      "amount": 0.11550151975683891, 
      "splitDirection": 2
    }, 
    "1": {
      "widget": {
        "applets": [
          {
            "ep": "oalab.applet", 
            "interface": "IPluginInstance", 
            "name": "ContextualMenu", 
            "properties": {
              "style": 0, 
              "title": False, 
              "toolbar": False
            }
          }
        ], 
        "interface": "IAppletContainer", 
        "properties": {
          "position": 0
        }
      }
    }, 
    "2": {
      "amount": 0.1609375, 
      "splitDirection": 1
    }, 
    "3": {
      "amount": 0.6190476190476191, 
      "splitDirection": 2
    }, 
    "4": {
      "amount": 0.49238095238095236, 
      "splitDirection": 1
    }, 
    "5": {
      "widget": {
        "applets": [
          {
            "ep": "oalab.applet", 
            "interface": "IPluginInstance", 
            "name": "ProjectManager", 
            "properties": {
              "title": False, 
              "toolbar": False
            }
          }
        ], 
        "interface": "IAppletContainer", 
        "properties": {
          "position": 0, 
          "title": "<b>Project</b>"
        }
      }
    }, 
    "6": {
      "widget": {
        "applets": [
          {
            "ep": "oalab.applet", 
            "interface": "IPluginInstance", 
            "name": "ControlManager", 
            "properties": {
              "icon": None, 
              "position": 2, 
              "state": "S'\\x00\\x00\\x00\\xff\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\xc0\\x01\\x00\\x00\\x00\\x07\\x01\\x00\\x00\\x00\\x02'\np0\n.", 
              "title": False, 
              "toolbar": False
            }
          }, 
          {
            "ep": "oalab.applet", 
            "interface": "IPluginInstance", 
            "name": "World", 
            "properties": {
              "title": False, 
              "toolbar": False
            }
          }, 
          {
            "ep": "oalab.applet", 
            "interface": "IPluginInstance", 
            "name": "PkgManagerWidget", 
            "properties": {
              "title": False, 
              "toolbar": False
            }
          }
        ], 
        "interface": "IAppletContainer", 
        "properties": {
          "position": 0
        }
      }
    }, 
    "7": {
      "amount": 0.8494809688581315, 
      "splitDirection": 2
    }, 
    "8": {
      "amount": 0.99812382739212, 
      "splitDirection": 2
    }, 
    "9": {
      "widget": {
        "applets": [
          {
            "ep": "oalab.applet", 
            "interface": "IPluginInstance", 
            "name": "FigureWidget", 
            "properties": {
              "num": 1, 
              "title": False, 
              "toolbar": False
            }
          }, 
          {
            "ep": "oalab.applet", 
            "interface": "IPluginInstance", 
            "name": "FigureWidget", 
            "properties": {
              "num": 0, 
              "title": False, 
              "toolbar": False
            }
          }
        ], 
        "interface": "IAppletContainer", 
        "properties": {
          "position": 0, 
          "title": "<b>2D</b> Viewers"
        }
      }
    }, 
    "10": {
      "widget": {
        "applets": [
          {
            "ep": "oalab.applet", 
            "interface": "IPluginInstance", 
            "name": "World", 
            "properties": {
              "title": False, 
              "toolbar": False
            }
          }
        ], 
        "interface": "IAppletContainer", 
        "properties": {
          "position": 2, 
          "title": "<b>3D</b> Viewers"
        }
      }
    }, 
    "11": {
      "widget": {
        "applets": [
          {
            "ep": "oalab.applet", 
            "interface": "IPluginInstance", 
            "name": "EditorManager", 
            "properties": {
              "title": False, 
              "toolbar": False
            }
          }
        ], 
        "interface": "IAppletContainer", 
        "properties": {
          "position": 0
        }
      }
    }, 
    "12": {
      "widget": {
        "applets": [
          {
            "ep": "oalab.applet", 
            "interface": "IPluginInstance", 
            "name": "ShellWidget", 
            "properties": {
              "title": False, 
              "toolbar": False
            }
          }, 
          {
            "ep": "oalab.applet", 
            "interface": "IPluginInstance", 
            "name": "HistoryWidget", 
            "properties": {
              "title": False, 
              "toolbar": False
            }
          }, 
          {
            "ep": "oalab.applet", 
            "interface": "IPluginInstance", 
            "name": "HelpWidget", 
            "properties": {
              "title": False, 
              "toolbar": False
            }
          }, 
          {
            "ep": "oalab.applet", 
            "interface": "IPluginInstance", 
            "name": "Logger", 
            "properties": {
              "title": False, 
              "toolbar": False
            }
          }
        ], 
        "interface": "IAppletContainer", 
        "properties": {
          "position": 2
        }
      }
    }
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
