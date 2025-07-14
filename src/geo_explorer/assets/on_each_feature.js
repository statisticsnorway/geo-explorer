window.onEachFeatureToggleHighlight = Object.assign({}, window.dashExtensions, {
    default: {
        popup: function (feature, layer) {
            window.selectedFeatureIds = window.selectedFeatureIds || [];
            const featureId = feature.properties._unique_id // || feature.properties.id || feature.properties.uuid;
            layer.clickCount = 0;

            // // Always check if feature is selected and apply highlight
            // if (window.selectedFeatureIds.includes(featureId)) {
            //     layer.setStyle({
            //         fillColor: "#ffff00",
            //         color: "#000000",
            //         weight: 3,
            //         fillOpacity: 1
            //     });
            // }

            // Store original style ON THE LAYER
            const originalStyle = {
                color: layer.options.color,
                fillColor: layer.options.fillColor,
                weight: layer.options.weight,
                fillOpacity: layer.options.fillOpacity
            };

            layer.on("click", function () {
                layer.clickCount += 1;
                const isOdd = layer.clickCount % 2 === 1;
                const idx = window.selectedFeatureIds.indexOf(featureId);
                if (isOdd & idx === -1) {
                    window.selectedFeatureIds.push(featureId);
                    layer.setStyle({
                        fillColor: "#ffff00",
                        color: "#000000",
                        weight: 3,
                        fillOpacity: 1
                    });
                } else {
                    window.selectedFeatureIds.splice(idx, 1);
                    layer.setStyle(originalStyle);
                }
            });
        }
    }
});