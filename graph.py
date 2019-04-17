# -*- coding: utf-8 -*-
import numpy as np

class LineGraph:
    def __init__(self, vis, lines, xlabel, ylabel, title, showlegend = False, display = -1):
        self.vis = vis
        self.showlegend = showlegend
        self.display = display
        self.x_data = {}
        self.y_data = {}
        
        for i in lines:
            self.x_data[i] = np.array([])
            self.y_data[i] = np.array([])
        
        self.plot = vis.line(X = np.array([np.nan]), 
                             Y = np.array([np.nan]), 
                             name = lines[0], 
                             opts=dict(xlabel = xlabel, 
                                       ylabel = ylabel, 
                                       title = title, 
                                       showlegend = showlegend))

        for i in range(1, len(lines)):
            vis.line(X = np.array([np.nan]), Y = np.array([np.nan]), name = lines[i], win = self.plot, update = "append")
    
    def update(self, line, x_value, y_value):
        self.x_data[line] = np.append(self.x_data[line], x_value)
        self.y_data[line] = np.append(self.y_data[line], y_value)
        
        if self.display != -1 and len(self.y_data[line]) == self.display:
            self.x_data[line] = np.delete(self.x_data[line], 0)
            self.y_data[line] = np.delete(self.y_data[line], 0)
            
        self.vis.line(X = self.x_data[line], Y = self.y_data[line], name = line, update = "replace", win = self.plot)
