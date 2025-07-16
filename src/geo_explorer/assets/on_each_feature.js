window.onEachFeatureToggleHighlight = Object.assign({}, window.dashExtensions, {
    default: {
        yellowIfHighlighted: function (feature, layer) {
            window.selectedFeatureIds = window.selectedFeatureIds || [];
            const featureId = feature.properties._unique_id; 
            const geomType = feature.geometry.type;

            // Store original style ON THE LAYER
            layer._originalStyle = {
                color: layer.options.color,
                fillColor: layer.options.fillColor,
                weight: layer.options.weight,
                fillOpacity: layer.options.fillOpacity
            };

            // Always check if feature is selected and apply highlight
            if (window.selectedFeatureIds.includes(featureId)) {
                if (geomType === "LineString" || geomType === "MultiLineString" || geomType === "LinearRing") {
                    layer.setStyle({
                        fillColor: "#ffff00",
                        color: "#ffff00",
                        weight: 3,
                        fillOpacity: 1
                    });
                } else {
                    layer.setStyle({
                        fillColor: "#ffff00",
                        color: "#000000",
                        weight: 3,
                        fillOpacity: 1
                    });
                }
            }

            layer.on("click", function () {
                const idx = window.selectedFeatureIds.indexOf(featureId);
                if (idx === -1) {
                    window.selectedFeatureIds.push(featureId);
                    if (geomType === "LineString" || geomType === "MultiLineString" || geomType === "LinearRing") {
                        layer.setStyle({
                            fillColor: "#ffff00",
                            color: "#ffff00",
                            weight: 3,
                            fillOpacity: 1
                        });
                    } else {
                        layer.setStyle({
                            fillColor: "#ffff00",
                            color: "#000000",
                            weight: 3,
                            fillOpacity: 1
                        });
                    }
                } else {
                    window.selectedFeatureIds.splice(idx, 1);
                    layer.setStyle(layer._originalStyle);
                }
                console.log(window.selectedFeatureIds);
            });
        },
        pointToLayerCircle: function(feature, latlng, context) {
            const {min, max, colorscale, circleOptions, colorProp} = context.hideout;
            const csc = chroma.scale(colorscale).domain([min, max]);
            circleOptions.fillColor = csc(feature.properties[colorProp]);
            return L.circleMarker(latlng, circleOptions);
        }
    }
});
