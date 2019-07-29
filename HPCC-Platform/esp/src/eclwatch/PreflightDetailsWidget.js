define([
    "dojo/_base/declare",
    "dojo/_base/lang",
    "dojo/i18n",
    "dojo/i18n!./nls/hpcc",
    "dojo/_base/array",
    "dojo/dom-class",
    "dojo/dom-construct",
    "dojo/dom-geometry",
    "dojo/_base/window",

    "dijit/registry",
    "dijit/form/TextBox",

    "hpcc/GridDetailsWidget",
    "src/ESPUtil",
    "src/ESPPreflight",

], function (declare, lang, i18n, nlsHPCC, arrayUtil, domClass, domConstruct, domGeom, win,
    registry, TextBox,
    GridDetailsWidget, ESPUtil, ESPPreflight
    ) {
        return declare("PreflightDetailsWidget", [GridDetailsWidget], {
            i18n: nlsHPCC,

            preflightTab: null,
            preflightWidgetLoaded: false,
            gridTitle: nlsHPCC.title_PreflightResults,
            idProperty: "__hpcc_id",

            init: function (params, route) {
                if (this.inherited(arguments))
                    return;
                this.initalized = false;
                this.params = params;
                this.setColumns(params);
                this.refresh(params, route);
            },

            refresh: function (params, route) {
                route === "machines" ? this.refreshMachinesGrid(params) : this.refreshClusterGrid(params);
            },

            createGrid: function (domID) {
                var context = this;

                this.openButton = registry.byId(this.id + "Open");
                this.openButton.set("disabled", true);

                var retVal = new declare([ESPUtil.Grid(true, false, {rowsPerPage:1000}, true)])({
                    store: this.store,
                    columns: this.setColumns()
                }, domID);
                return retVal;
            },

            calculateColumnWidth: function (grid) {
                for (var key in grid.columns) {
                    console.log(grid.columns[key].id)
                    if (grid.columns[key].label !== "Location" && grid.columns[key].label !== "Component") { //this needs improvement
                        var node = domConstruct.toDom('<div style="position:absolute;visibility:hidden">' + grid.columns[key].label + '</div>');
                        domConstruct.place(node, win.body());
                        var p = domGeom.position(node);
                        domConstruct.destroy(node);
                        grid.resizeColumnWidth(grid.columns[key].id, (Math.ceil(p.w) + 20));
                    }
                }
            },

            setColumns: function (params) {
                var context = this;
                var dynamicColumns = {
                    Location: {label: this.i18n.Location, id: this.i18n.Location, width: 350},
                    Component: {label: this.i18n.Component, id: this.i18n.Component, width: 275},
                    ComputerUpTime: { label: this.i18n.ComputerUpTime, width: 75 }
                }

                function handleResponse(response, dynamicColumns) {
                    arrayUtil.forEach(response, function (row, idx) {
                        if (row.Storage) {
                            var request = params.RequestInfo.DiskThreshold;
                            arrayUtil.forEach(row.Storage.StorageInfo, function(storageitem, storageidx) {
                                var swap = storageitem.Description === "Swap"; //swap if === 0 should be N/A (data needs restructuring in HPCC-21667)
                                var tempObj =  {};
                                var cleanColumnName = context.cleanColumn(storageitem.Description);
                                tempObj[cleanColumnName] = {
                                    label: storageitem.Description,
                                    renderCell: function(object, value, node, options) {
                                        switch (request > value && value !== 0) {
                                            case true:
                                                domClass.add(node, "WarningCell");
                                            break;
                                        }
                                        if (swap && value !== 0) {
                                            node.innerText = value + "%" || "N/A"
                                        } else if (!value) {
                                            node.innerText = "";
                                        } else {
                                            node.innerText = value + "%";
                                        }
                                    }
                                }
                                lang.mixin(dynamicColumns, tempObj);
                            });
                        }
                    });
                    return dynamicColumns;
                }

                if (params) {
                    if (params.TargetClusterInfoList) {
                        handleResponse(params.TargetClusterInfoList.TargetClusterInfo[0].Processes.MachineInfoEx, dynamicColumns)
                    } else {
                        handleResponse(params.Machines.MachineInfoEx, dynamicColumns)
                    }

                    var finalColumns = params.Columns.Item;
                    for (var index in finalColumns) {
                        var clean = this.cleanColumn(finalColumns[index]);

                        if (clean === "Condition") {
                            lang.mixin(dynamicColumns, {
                                "Condition": {
                                    label: this.i18n.Condition,
                                    renderCell: function(object, value, node, options) {
                                        switch (value) {
                                            case 'Unknown':
                                            case 'Warning':
                                            case 'Minor':
                                            case 'Major':
                                            case 'Critical':
                                            case 'Fatal':
                                                domClass.add(node, "WarningCell");
                                            break;
                                        }
                                        node.innerText = value || "";
                                    }
                                }
                            });
                        }

                        if (clean === "State") {
                            lang.mixin(dynamicColumns, {
                                "State": {
                                    label: this.i18n.State,
                                    renderCell: function(object, value, node, options) {
                                        switch (value) {
                                            case 'Unknown':
                                            case 'Starting':
                                            case 'Stopping':
                                            case 'Suspended':
                                            case 'Recycling':
                                            case 'Busy':
                                            case 'NA':
                                                domClass.add(node, "WarningCell");
                                            break;
                                        }
                                        node.innerText = value || "";
                                    }
                                }
                            });
                        }

                        if (clean === "CPULoad") {
                            var request = params.RequestInfo.CpuThreshold;
                            lang.mixin(dynamicColumns, {
                                CPULoad: {
                                    label: this.i18n.CPULoad,
                                    renderCell: function (object, value, node, options) {
                                        switch (request < value) {
                                            case true:
                                                domClass.add(node, "WarningCell");
                                            break;
                                        }
                                        node.innerText = value + "%";
                                    }
                                }
                            });
                        }

                        if (clean === "RoxieState") {
                            lang.mixin(dynamicColumns, {
                                RoxieState: {
                                    label: this.i18n.RoxieState,
                                    renderCell: function (object, value, node, options) {
                                        switch (value) {
                                            case "State hash mismatch ...":
                                            case "Not attached to DALI...":
                                            case "empty state hash ...":
                                            case "Node State: not ok ...":
                                                domClass.add(node, "WarningCell");
                                            break;
                                        }
                                        node.innerText = value || "N/A";
                                    }
                                }
                            });
                        }
                    }
                     context.grid.set("columns", dynamicColumns);
                     context.calculateColumnWidth(context.grid);
                }
            },

            cleanColumn: function (str) {
                var clean = str.replace(/([~!@#$%^&*()_+=`{}\[\]\|\\:;'<>,.\/? ])+/g, '').replace(/^(-)+|(-)+$/g, '');
                return clean;
            },

            refreshMachinesGrid: function (params) {
                var context = this;
                var params = this.params;
                var results = [];

                if (params) {
                    arrayUtil.forEach(params.Machines.MachineInfoEx, function (row, idx) {
                        var dynamicRowsObj = {};
                        row.RoxieState ? lang.mixin(dynamicRowsObj, {
                            RoxieState: row.RoxieState
                        }) : "";
                        lang.mixin(dynamicRowsObj, {
                            Location: row.Address + " " + row.ComponentPath,
                            Component: row.DisplayType + "[" + row.ComponentName + "]",
                            ComputerUpTime: row.UpTime
                        });
                        if (row.Processors) {
                            arrayUtil.forEach(row.Processors.ProcessorInfo, function (processor, idx) {
                                lang.mixin(dynamicRowsObj, {
                                    CPULoad: processor.Load
                                });
                            });
                        }
                        if (row.Storage) {
                            arrayUtil.forEach(row.Storage.StorageInfo, function (storage, idx) {
                                var cleanColumn = context.cleanColumn(storage.Description);
                                var tmpObj = {};
                                tmpObj[cleanColumn] = storage.PercentAvail;
                                lang.mixin(dynamicRowsObj, tmpObj);
                            });
                        }
                        results.push(dynamicRowsObj);
                    });
                }

                context.store.setData(results);
                context.grid.set("query", {});
            },

            refreshClusterGrid: function (params) {
                var context = this;
                var params = this.params;
                var results = [];

                if (params) {
                    arrayUtil.forEach(params.TargetClusterInfoList.TargetClusterInfo, function (row, idx) {
                        arrayUtil.forEach(row.Processes.MachineInfoEx, function (setRows, idx) {
                            var dynamicRowsObj = {};
                            setRows.RoxieState ? lang.mixin(dynamicRowsObj, {
                                RoxieState: setRows.RoxieState
                            }) : "";
                            lang.mixin(dynamicRowsObj, {
                                Location: setRows.Address + " " + setRows.ComponentPath,
                                Component: setRows.DisplayType + "[" + setRows.ComponentName + "]",
                                ComputerUpTime: setRows.UpTime
                            });

                            if (setRows.ComponentInfo) {
                                lang.mixin(dynamicRowsObj, {
                                    Condition: ESPPreflight.getCondition(setRows.ComponentInfo.Condition),
                                    State: ESPPreflight.getState(setRows.ComponentInfo.State),
                                    UpTime: setRows.ComponentInfo.UpTime
                                });
                            }
                            if (setRows.Processors) {
                                arrayUtil.forEach(setRows.Processors.ProcessorInfo, function (processor, idx) {
                                    lang.mixin(dynamicRowsObj, {
                                        CPULoad: processor.Load
                                    });
                                });
                            }
                            if (setRows.Storage) {
                                arrayUtil.forEach(setRows.Storage.StorageInfo, function (storage, idx) {
                                    var cleanColumn = context.cleanColumn(storage.Description)
                                    var tmpObj = {};
                                    tmpObj[cleanColumn] = storage.PercentAvail
                                    lang.mixin(dynamicRowsObj, tmpObj);
                                });
                            }
                            results.push(dynamicRowsObj);
                        });
                    });
                }
                context.store.setData(results);
                context.grid.set("query", {});
            }
        });
    });