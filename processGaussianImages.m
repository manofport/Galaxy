function [X] = processGaussianImages(img_dir,imagelist,imagewidth,p_shrink_factor,p_imgtrim,p_images_to_process,results_dir)

    m = size(imagelist,1);

  
    imgleft = p_imgtrim+1;
    imgright = imagewidth-p_imgtrim;
    imgtop = p_imgtrim+1;
    imgbottom = imagewidth-p_imgtrim;

    %fprintf("Pre-processing images.\n");
    %newtest
    checkpoint = 100;
    check = 1;
    imgvec = [];
    checkvec = [];

    m=424-2*p_imgtrim;
    n=m;
    step = p_shrink_factor;

    %outimg = img(1:step:m,1:step:n);

    rowarray = [];
    colarray = [];
    for i = 1:m
        if rem(i,step) > 0
            rowarray = [rowarray i];
        endif
    endfor


    for i = 1:n
        if rem(i,step) > 0
            colarray = [colarray i];
        endif
    endfor




    for i = 1:p_images_to_process
        image = imagelist{i};
        [img,map,alpha] = imread(image);



        trimmedimg = img(imgtop:imgbottom,imgleft:imgright,:);
        size(trimmedimg);

        smallimg = GaussianBlur(trimmedimg,4,8,1);
        %imshow(smallimg)
        %pause

        %subplot(1,3,1), imshow(img); subplot(1,3,2), imshow(trimmedimg); subplot(1,3,3), imshow(smallimg);
        %pause;



        %trim the image
        %trimmedimg = img(imgtop:imgbottom,imgleft:imgright,:);
        %smallimg = shrinkColorImg(trimmedimg, p_shrink_factor,rowarray,colarray);
        grayimg = flipud(convertToGray(smallimg)); %need to flip to maintain order
        imgvec = [grayimg(:)'; imgvec];

%        tempX(:,i) = double(imgvec);
        if check == checkpoint
            checkvec = [imgvec; checkvec];
            imgvec = [];
            curtime = sprintf("%d",clock);
            fprintf("Processed %d images\n%s\n",i,curtime);
            fflush(stdout);
            check = 1;
        else
            check++;
        endif
    endfor
    
    checkvec = [imgvec; checkvec];
    X = flipud(double(checkvec));%flip back to get original order!

end

